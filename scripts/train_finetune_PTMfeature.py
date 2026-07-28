import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy


class EarlyStoppingPatched(EarlyStopping):
    """Keep CLI patience when loading callback state from a checkpoint."""

    def load_state_dict(self, state_dict):
        saved_patience = self.patience
        super().load_state_dict(state_dict)
        self.patience = saved_patience

try:
    from swanlab.integration.pytorch_lightning import SwanLabLogger
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False

from trainer.config import parse_train_args
from trainer.module import ProteinWrapper
from data.module import DataModule
from trainer.checkpoint import (
    smart_load_checkpoint, perform_embedding_surgery, warmstart_ptm_tokens, save_args,
)


def audit_finetune_mode(args, model):
    """Fail early if a launcher accidentally selects the wrong fine-tune mode."""
    actual_mode = "lora" if getattr(args, "use_lora", False) else "full"
    expected_mode = getattr(args, "expected_finetune_mode", "auto")
    if expected_mode != "auto" and expected_mode != actual_mode:
        raise ValueError(
            f"Fine-tune mode mismatch: expected {expected_mode}, got {actual_mode}. "
            f"Check whether --use_lora is present."
        )

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    fraction = 100.0 * trainable / max(total, 1)
    print("=" * 72)
    print(f"Fine-tune mode : {actual_mode}")
    print(f"Total params   : {total / 1e6:.3f}M")
    print(f"Trainable      : {trainable / 1e6:.3f}M ({fraction:.2f}%)")
    print(f"PTM channel    : {bool(getattr(args, 'use_ptm_channel', False))}")
    print("=" * 72)

    if actual_mode == "full" and fraction < 90.0:
        raise ValueError(
            f"Full fine-tuning requested, but only {fraction:.2f}% of parameters "
            "are trainable."
        )
    if actual_mode == "lora" and fraction >= 90.0:
        raise ValueError(
            f"LoRA fine-tuning requested, but {fraction:.2f}% of parameters are "
            "trainable."
        )


def main():
    args = parse_train_args()

    if args.ptm_feat_path:
        print(f"PTM feature path: {args.ptm_feat_path}")

    if args.ckpt is None:
        raise ValueError("--ckpt is required.")

    save_args(args, os.path.join(os.environ["MODEL_DIR"], 'args'))
    print(f"MODEL_DIR: {os.environ['MODEL_DIR']}")
    pl.seed_everything(42)

    dm = DataModule(args)
    model = ProteinWrapper(args)
    model = smart_load_checkpoint(model, args.ckpt)
    audit_finetune_mode(args, model)

    if args.ema and hasattr(model, 'ema'):
        print("Syncing pretrained weights into EMA shadow weights...")
        model.ema.load_state_dict({
            "params": model.model.state_dict(),
            "decay": model.ema.decay,
            "num_updates": getattr(model.ema, "num_updates", 0),
        })

    # With the explicit PTM channel as the carrier, the token-surgery route
    # (expanding aa embedding for SEP/TPO/PTR) is redundant — skip it.
    if not getattr(args, 'use_ptm_channel', False):
        model = perform_embedding_surgery(model)
        # For bases like aa_multi the aa embedding is already 24, so surgery is skipped
        # → the phospho token rows (21/22/23) stay at random init. Copy them from the
        # parent residues S/T/Y (matching the old AA scheme).
        if getattr(args, 'warmstart_ptm_tokens', False):
            model = warmstart_ptm_tokens(model)
    else:
        print("Using explicit PTM channel; skipping embedding surgery.")

    logger = None
    if args.swanlab and SWANLAB_AVAILABLE:
        logger = SwanLabLogger(
            project="PTM_Finetune",
            experiment_name=args.run_name,
            config=vars(args),
            save_dir="./swanlab_logs",
        )
    elif args.swanlab:
        print("Warning: --swanlab flag set but swanlab is not installed. Logging disabled.")

    ckpt_dir = os.path.join(os.environ["MODEL_DIR"], 'checkpoints')
    checkpoint_periodic = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename='backup-epoch={epoch:03d}-val_loss={val_loss:.4f}',
        auto_insert_metric_name=False,
        save_top_k=-1,
        every_n_epochs=args.ckpt_freq,
        save_weights_only=False,
    )
    callbacks = [checkpoint_periodic]

    if getattr(args, 'early_stop_patience', 0) > 0 and not args.no_validate:
        callbacks.append(EarlyStoppingPatched(
            monitor='val_loss',
            patience=args.early_stop_patience,
            min_delta=getattr(args, 'early_stop_min_delta', 1e-4),
            mode='min',
            verbose=True,
        ))

    # Save the best checkpoint by val_loss (mirror train.py) — avoids the
    # "only backup checkpoints" problem that bit esm2_multi.
    if not getattr(args, 'no_validate', False):
        checkpoint_best = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename='best-epoch={epoch:03d}-val_loss={val_loss:.4f}',
            auto_insert_metric_name=False,
            monitor='val_loss',
            mode='min',
            save_top_k=1,
            save_weights_only=False,
            save_on_train_epoch_end=False,
            verbose=True,
        )
        callbacks.append(checkpoint_best)

    has_esm2 = (getattr(args, 'esm2_emb_path', None) is not None
                or bool(getattr(args, 'esm_paths', None)))
    strategy = DDPStrategy(find_unused_parameters=True) if has_esm2 else 'ddp'

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else 'auto',
        strategy=strategy,
        max_epochs=args.epochs,
        limit_train_batches=args.train_batches or 1.0,
        limit_val_batches=0.0 if args.no_validate else (args.val_batches or 1.0),
        num_sanity_val_steps=0,
        precision=args.precision,
        enable_progress_bar=not args.swanlab,
        gradient_clip_val=args.grad_clip,
        default_root_dir=os.environ["MODEL_DIR"],
        callbacks=callbacks,
        accumulate_grad_batches=args.accumulate_grad,
        val_check_interval=args.val_freq,
        check_val_every_n_epoch=args.val_epoch_freq,
        logger=logger if logger is not None else False,
    )

    print("Starting fine-tuning...")
    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    main()

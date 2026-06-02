import os
import argparse
import torch
import pytorch_lightning as pl

# PyTorch 2.6 changed torch.load default to weights_only=True.
# PL checkpoints contain argparse.Namespace (from save_hyperparameters),
# so we must allowlist it before any checkpoint is loaded.
torch.serialization.add_safe_globals([argparse.Namespace])
from pytorch_lightning.callbacks import ModelCheckpoint

try:
    from swanlab.integration.pytorch_lightning import SwanLabLogger
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False

from trainer.config import parse_train_args


def main():
    # parse_train_args() sets MODEL_DIR in os.environ.
    # Modules that call get_logger() at import time must be imported AFTER this
    # so the logger's FileHandler writes to the correct directory.
    args = parse_train_args()

    from trainer.module import ProteinWrapper
    from data.module import DataModule
    from trainer.checkpoint import smart_load_checkpoint, save_args

    save_args(args, os.path.join(os.environ["MODEL_DIR"], 'args'))
    print(f"MODEL_DIR: {os.environ['MODEL_DIR']}")
    pl.seed_everything(42)

    dm = DataModule(args)
    model = ProteinWrapper(args)

    resume_ckpt = None
    if getattr(args, 'resume', False) and args.ckpt is not None:
        # Resume: restore full training state (epoch, optimizer, scheduler)
        print(f"Resuming training from: {args.ckpt}")
        resume_ckpt = args.ckpt
    elif args.ckpt is not None:
        # Fine-tune: load weights only, start from epoch 0
        print(f"Loading checkpoint weights: {args.ckpt}")
        model = smart_load_checkpoint(model, args.ckpt)
    else:
        print("No checkpoint provided — training from scratch.")

    if not resume_ckpt and args.ema and hasattr(model, 'ema'):
        print("Syncing pretrained weights into EMA shadow weights...")
        model.ema.load_state_dict({
            "params": model.model.state_dict(),
            "decay": model.ema.decay,
            "num_updates": getattr(model.ema, "num_updates", 0),
        })

    logger = None
    if args.swanlab and SWANLAB_AVAILABLE:
        logger = SwanLabLogger(
            project="ProteinFlux_Pretrain",
            experiment_name=args.run_name,
            config=vars(args),
            save_dir="./swanlab_logs",
        )
    elif args.swanlab:
        print("Warning: --swanlab flag set but swanlab is not installed. Logging disabled.")

    ckpt_dir = os.path.join(os.environ["MODEL_DIR"], 'checkpoints')

    checkpoint_periodic = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename='backup_{epoch:02d}',
        save_top_k=-1,
        every_n_epochs=args.ckpt_freq,
        save_weights_only=False,
    )

    callbacks = [checkpoint_periodic]

    if not args.no_validate:
        checkpoint_best = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename='best',
            monitor='val_loss',
            mode='min',
            save_top_k=1,
            save_weights_only=False,
            save_on_train_epoch_end=False,
            verbose=True,
        )
        callbacks.append(checkpoint_best)

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else 'auto',
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

    print("Starting training...")
    trainer.fit(model, datamodule=dm, ckpt_path=resume_ckpt)


if __name__ == '__main__':
    main()

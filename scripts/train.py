import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

try:
    from swanlab.integration.pytorch_lightning import SwanLabLogger
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False

from trainer.config import parse_train_args
from trainer.module import ProteinWrapper
from data.module import DataModule
from trainer.checkpoint import smart_load_checkpoint, save_args


def main():
    args = parse_train_args()

    save_args(args, os.path.join(os.environ["MODEL_DIR"], 'args'))
    print(f"MODEL_DIR: {os.environ['MODEL_DIR']}")
    pl.seed_everything(42)

    dm = DataModule(args)
    model = ProteinWrapper(args)

    if args.ckpt is not None:
        print(f"Loading checkpoint: {args.ckpt}")
        model = smart_load_checkpoint(model, args.ckpt)
    else:
        print("No checkpoint provided — training from scratch.")

    if args.ema and hasattr(model, 'ema'):
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

    checkpoint_periodic = ModelCheckpoint(
        dirpath=os.path.join(os.environ["MODEL_DIR"], 'checkpoints'),
        filename='backup_{epoch:02d}',
        save_top_k=-1,
        every_n_epochs=args.ckpt_freq,
        save_weights_only=False,
    )

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
        callbacks=[checkpoint_periodic],
        accumulate_grad_batches=args.accumulate_grad,
        val_check_interval=args.val_freq,
        check_val_every_n_epoch=args.val_epoch_freq,
        logger=logger,
    )

    print("Starting training...")
    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    main()

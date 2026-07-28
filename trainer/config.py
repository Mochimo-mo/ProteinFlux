from argparse import ArgumentParser
import os


def parse_train_args():
    parser = ArgumentParser()

    ## Trainer settings
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--resume", action='store_true', default=False,
                        help='Resume training from --ckpt (restores epoch, optimizer, scheduler). '
                             'Without this flag, --ckpt only loads model weights (fine-tuning).')
    parser.add_argument("--validate", action='store_true', default=False)
    parser.add_argument("--num_workers", type=int, default=4)

    ## Epoch settings
    group = parser.add_argument_group("Epoch settings")
    group.add_argument("--epochs", type=int, default=100)
    group.add_argument("--overfit", action='store_true')
    group.add_argument("--overfit_peptide", type=str, default=None)
    group.add_argument("--overfit_frame", action='store_true')
    group.add_argument("--train_batches", type=int, default=None)
    group.add_argument("--val_batches", type=int, default=None)
    group.add_argument("--inference_batches", type=int, default=0)
    group.add_argument("--batch_size", type=int, default=1)
    group.add_argument("--val_freq", type=int, default=None)
    group.add_argument("--val_epoch_freq", type=int, default=1)
    group.add_argument("--no_validate", action='store_true')
    group.add_argument("--early_stop_patience", type=int, default=0,
                       help="Stop if val_loss does not improve for N epochs. 0 = disabled.")
    group.add_argument("--early_stop_min_delta", type=float, default=1e-4,
                       help="Minimum improvement in val_loss to count as progress.")
    group.add_argument("--designability_freq", type=int, default=1)

    ## Logging args
    group = parser.add_argument_group("Logging settings")
    group.add_argument("--print_freq", type=int, default=100)
    group.add_argument("--ckpt_freq", type=int, default=10)
    group.add_argument("--run_name", type=str, default="default")
    parser.add_argument('--swanlab', action='store_true', help='Use SwanLab for logging')

    ## Optimization settings
    group = parser.add_argument_group("Optimization settings")
    group.add_argument("--accumulate_grad", type=int, default=1)
    group.add_argument("--grad_clip", type=float, default=1.)
    group.add_argument("--check_grad", action='store_true')
    group.add_argument('--grad_checkpointing', action='store_true')
    group.add_argument('--adamW', action='store_true')
    group.add_argument('--ema', action='store_true')
    group.add_argument('--ema_decay', type=float, default=0.999)
    group.add_argument("--lr", type=float, default=1e-4)
    group.add_argument("--min_lr", type=float, default=1e-6,
                       help="Minimum LR at the end of cosine decay.")
    group.add_argument("--warmup_epochs", type=int, default=0,
                       help="Linear warmup epochs. 0 = no warmup (constant LR).")
    group.add_argument('--precision', type=str, default='32-true')

    ## Training data
    group = parser.add_argument_group("Training data settings")
    group.add_argument('--train_split', type=str, default=None)
    group.add_argument('--val_split', type=str, default=None)
    group.add_argument('--data_dir', type=str, default=None)
    group.add_argument('--num_frames', type=int, default=50)
    group.add_argument('--crop', type=int, default=256)
    group.add_argument('--repeat', type=int, default=20)
    group.add_argument('--copy_frames', action='store_true')
    group.add_argument('--max_train_frames', type=int, default=None,
                       help='Truncate each trajectory to the first N frames before sampling. '
                            'e.g. 100 = use only first 10ns when interval is 100ps.')
    group.add_argument('--fixed_ptm_crop', action='store_true',
                       help='Whether to use PTM-centered fixed window cropping')
    parser.add_argument('--ptm_feat_path', type=str, default=None,
                        help='Path to the PTM features pickle file')
    parser.add_argument('--ptm_feat_dim', type=int, default=256,
                        help='Dimension of residue-level PTM features. '
                             'Use 512 for FluxSite sequence-tower features.')
    group.add_argument('--use_ptm_channel', action='store_true',
                       help='Explicit PTM-type conditioning channel (pSer/pThr/pTyr), '
                            'derived from seqres tokens 21/22/23; zero-init, base-safe.')
    group.add_argument('--num_ptm_types', type=int, default=3,
                       help='Number of PTM identities (3 = pSer/pThr/pTyr).')
    group.add_argument('--warmstart_ptm_tokens', action='store_true',
                       help='Warm-start aa-embedding PTM rows 21/22/23 from parent '
                            'S/T/Y (15/16/18). For AA-base token route when the '
                            'embedding is already 24 and surgery is skipped.')

    ## LoRA finetune settings
    group = parser.add_argument_group("LoRA settings")
    group.add_argument('--use_lora', action='store_true',
                       help='Freeze backbone, LoRA-adapt attn/FFN, fully train PTM channels.')
    group.add_argument(
        '--expected_finetune_mode',
        choices=['auto', 'lora', 'full'],
        default='auto',
        help=(
            'Safety assertion for PTM fine-tuning launchers. '
            'lora requires --use_lora; full requires it to be absent.'
        ),
    )
    group.add_argument('--lora_r', type=int, default=8)
    group.add_argument('--lora_alpha', type=int, default=16)
    group.add_argument('--lora_dropout', type=float, default=0.05)

    ## ESM2 embedding settings
    group = parser.add_argument_group("ESM2 embedding settings")
    group.add_argument('--esm2_emb_path', type=str, default=None,
                       help='Path to pre-computed ESM2 embeddings pkl file. '
                            'Expected format: {protein_name: np.ndarray [L, esm2_dim]}. '
                            'When set, ESM2 embeddings are used as an additional residue-level condition.')
    group.add_argument('--esm2_dim', type=int, default=1280,
                       help='Dimension of ESM2 embeddings (1280 for esm2_t33_650M, 480 for esm2_t12_35M)')
    group.add_argument('--esm2_proj_dim', type=int, default=None,
                       help='Hidden dim of the two-layer ESM2 projection MLP. '
                            'Defaults to (esm2_dim + embed_dim) // 2. '
                            'Set to 0 to use a single linear layer instead.')

    ### Masking settings
    group = parser.add_argument_group("Masking settings")
    group.add_argument('--no_aa_emb', action='store_true')

    ## Model settings
    group = parser.add_argument_group("Model settings")
    group.add_argument('--hyena', action='store_true')
    group.add_argument('--no_rope', action='store_true')
    group.add_argument('--dropout', type=float, default=0.0)
    group.add_argument('--interleave_ipa', action='store_true')
    group.add_argument('--prepend_ipa', action='store_true')
    group.add_argument('--oracle', action='store_true')
    group.add_argument('--num_layers', type=int, default=5)
    group.add_argument('--embed_dim', type=int, default=384)
    group.add_argument('--mha_heads', type=int, default=16)
    group.add_argument('--ipa_heads', type=int, default=4)
    group.add_argument('--ipa_head_dim', type=int, default=32)
    group.add_argument('--ipa_qk', type=int, default=8)
    group.add_argument('--ipa_v', type=int, default=8)
    group.add_argument('--time_multiplier', type=float, default=100.)
    group.add_argument('--abs_pos_emb', action='store_true')
    group.add_argument('--abs_time_emb', action='store_true')

    group = parser.add_argument_group("Transport arguments")
    group.add_argument("--path-type", type=str, default="GVP", choices=["Linear", "GVP", "VP"])
    group.add_argument("--prediction", type=str, default="velocity",
                       choices=["velocity", "score", "noise"])
    group.add_argument("--sampling_method", type=str, default="dopri5",
                       choices=["dopri5", "euler"])

    group = parser.add_argument_group("Multi-dataset training (ATLAS + MoDEL …)")
    group.add_argument('--multi_dataset', action='store_true',
                       help="Enable mixed multi-dataset training (uses the *_data_dir / *_train_split args below)")
    group.add_argument('--atlas_data_dir', type=str, default=None)
    group.add_argument('--model_data_dir', type=str, default=None)
    group.add_argument('--atlas_train_split', type=str, default=None)
    group.add_argument('--atlas_val_split',   type=str, default=None)
    group.add_argument('--model_train_split', type=str, default=None)
    group.add_argument('--model_val_split',   type=str, default=None)
    group.add_argument('--dataset_weights', type=str, default=None,
                       help='JSON, e.g. {"atlas":0.5,"model":0.5} (per-dataset sampling weights)')
    group.add_argument('--dataset_frame_intervals', type=str, default=None,
                       help='JSON, e.g. {"atlas":1,"model":1} (per-dataset downsampling stride)')
    group.add_argument('--esm_paths', type=str, default=None,
                       help='JSON, e.g. {"atlas":"a.pkl","model":"b.pkl"} (per-dataset ESM2 pkl)')
    group.add_argument('--max_resample', type=int, default=20,
                       help='Max resampling attempts when a trajectory is too short / ESM is missing')

    group = parser.add_argument_group("Ensemble distribution-matching loss")
    group.add_argument("--ensemble_loss", action="store_true",
                       help="Enable ensemble-level distribution-matching loss (trained jointly with flow loss)")
    group.add_argument("--ensemble_kind", type=str, default="energy",
                       choices=["energy", "mmd", "sw"])
    group.add_argument("--ensemble_feature", type=str, default="per_residue",
                       choices=["per_residue", "global"])
    group.add_argument("--ensemble_lambda", type=float, default=0.1,
                       help="Weight of the distribution loss")
    group.add_argument("--ensemble_warmup", type=int, default=20,
                       help="Number of epochs to linearly ramp λ from 0 to its target value")
    group.add_argument("--ensemble_tau_min", type=float, default=0.3,
                       help="Skip when flow time t<tau_min (endpoint prediction too coarse)")


    args = parser.parse_args()
    os.environ["MODEL_DIR"] = os.path.join("workdir", args.run_name)

    return args

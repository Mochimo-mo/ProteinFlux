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
    group.add_argument('--fixed_ptm_crop', action='store_true',
                       help='Whether to use PTM-centered fixed window cropping')
    parser.add_argument('--ptm_feat_path', type=str, default=None,
                        help='Path to the PTM features pickle file')

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


    args = parser.parse_args()
    os.environ["MODEL_DIR"] = os.path.join("workdir", args.run_name)

    return args

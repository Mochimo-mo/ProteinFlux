import torch.utils.data
import pytorch_lightning as pl

from data.trajectory import ProteinDataset, PTMDataset, ESM2Dataset, MultiProteinDataset


def _select_dataset_cls(args):
    """Pick the dataset class based on which feature paths are provided.

    Priority: PTMDataset > ESM2Dataset > ProteinDataset
    """
    if getattr(args, 'ptm_feat_path', None):
        return PTMDataset
    if getattr(args, 'esm2_emb_path', None):
        return ESM2Dataset
    return ProteinDataset


class DataModule(pl.LightningDataModule):
    """Lightning DataModule that selects the dataset class based on available feature paths."""

    def __init__(self, args):
        super().__init__()
        self.args = args

    def setup(self, stage=None):
        if getattr(self.args, 'multi_dataset', False):
            # Mixed multi-dataset training (ATLAS + MoDEL ...)
            train_files, val_files = {}, {}
            if getattr(self.args, 'atlas_train_split', None):
                train_files['atlas'] = self.args.atlas_train_split
                val_files['atlas']   = getattr(self.args, 'atlas_val_split', None)
            if getattr(self.args, 'model_train_split', None):
                train_files['model'] = self.args.model_train_split
                val_files['model']   = getattr(self.args, 'model_val_split', None)
            val_files = {k: v for k, v in val_files.items() if v}
            self.train_ds = MultiProteinDataset(self.args, train_files, repeat=self.args.repeat)
            self.val_ds   = MultiProteinDataset(self.args, val_files,   repeat=1)
            return
        dataset_cls = _select_dataset_cls(self.args)
        self.train_ds = dataset_cls(self.args, self.args.train_split, repeat=self.args.repeat)
        self.val_ds = dataset_cls(self.args, self.args.val_split, repeat=1)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds, batch_size=self.args.batch_size,
            shuffle=True, num_workers=self.args.num_workers,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds, batch_size=self.args.batch_size,
            shuffle=False, num_workers=self.args.num_workers,
        )

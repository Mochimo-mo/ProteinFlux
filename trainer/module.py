from trainer.ema import ExponentialMovingAverage
from core.logger import get_logger

logger = get_logger(__name__)

import pytorch_lightning as pl
import torch, time, os
import torch.nn as nn
import numpy as np
import pandas as pd
from core.rigid_utils import Rigid, Rotation
from collections import defaultdict
from functools import partial

from model.backbone import LatentModel
from flow.matching import create_transport, Sampler
from core.flow_utils import get_offsets
from core.io_utils import atom14_to_pdb
from core.tensor_utils import tensor_tree_map
from core.geometry import frames_torsions_to_atom14
from core.ptm_utils import map_ptm_to_parent

try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False


def gather_log(log, world_size):
    if world_size == 1:
        return log
    log_list = [None] * world_size
    torch.distributed.all_gather_object(log_list, log)
    log = {key: sum([l[key] for l in log_list], []) for key in log}
    return log


def get_log_mean(log):
    out = {}
    for key in log:
        try:
            out[key] = np.nanmean(log[key])
        except:
            pass
    return out


class Wrapper(pl.LightningModule):

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self._log = defaultdict(list)
        self.last_log_time = time.time()
        self.iter_step = 0

    def log(self, key, data):
        if isinstance(data, torch.Tensor):
            data = data.mean().item()
        log = self._log
        if self.stage == 'train' or self.args.validate:
            log["iter_" + key].append(data)
        log[self.stage + "_" + key].append(data)

    def load_ema_weights(self):
        logger.info('Loading EMA weights')
        clone_param = lambda t: t.detach().clone()
        self.cached_weights = tensor_tree_map(clone_param, self.model.state_dict())
        self.model.load_state_dict(self.ema.state_dict()["params"])

    def restore_cached_weights(self):
        logger.info('Restoring cached weights')
        self.model.load_state_dict(self.cached_weights)
        self.cached_weights = None

    def on_before_zero_grad(self, *args, **kwargs):
        if self.args.ema:
            self.ema.update(self.model)

    def training_step(self, batch, batch_idx):
        if self.args.ema:
            if (self.ema.device != self.device):
                self.ema.to(self.device)
        return self.general_step(batch, stage='train')

    def validation_step(self, batch, batch_idx):
        if self.args.ema:
            if (self.ema.device != self.device):
                self.ema.to(self.device)
            if (self.cached_weights is None):
                self.load_ema_weights()

        self.general_step(batch, stage='val')
        self.validation_step_extra(batch, batch_idx)
        if self.args.validate and self.iter_step % self.args.print_freq == 0:
            self.print_log()

    def validation_step_extra(self, batch, batch_idx):
        pass

    def on_train_epoch_end(self):
        self.print_log(prefix='train', save=False)

    def on_validation_epoch_end(self):
        if self.args.ema:
            self.restore_cached_weights()

        # Capture val_loss BEFORE print_log deletes the val_ entries from self._log
        val_loss_list = self._log.get('val_loss', [])
        if val_loss_list:
            mean_val_loss = float(np.nanmean(val_loss_list))
            super().log('val_loss', mean_val_loss, sync_dist=True, prog_bar=True)

        self.print_log(prefix='val', save=False)

    def on_before_optimizer_step(self, optimizer):
        if (self.trainer.global_step + 1) % self.args.print_freq == 0:
            self.print_log()

        if self.args.check_grad:
            for name, p in self.model.named_parameters():
                if p.grad is None:
                    logger.warning(f"Param {name} has no grad")

    def on_load_checkpoint(self, checkpoint):
        logger.info('Loading EMA state dict')
        if self.args.ema:
            ema = checkpoint["ema"]
            self.ema.load_state_dict(ema)

    def on_save_checkpoint(self, checkpoint):
        if self.args.ema:
            if self.cached_weights is not None:
                self.restore_cached_weights()
            checkpoint["ema"] = self.ema.state_dict()

    def print_log(self, prefix='iter', save=False, extra_logs=None):
        log = self._log
        log = {key: log[key] for key in log if f"{prefix}_" in key}
        log = gather_log(log, self.trainer.world_size)
        mean_log = get_log_mean(log)

        mean_log.update({
            'epoch': self.trainer.current_epoch,
            'trainer_step': self.trainer.global_step + int(prefix == 'iter'),
            'iter_step': self.iter_step,
            f'{prefix}_count': len(log[next(iter(log))]),
        })
        if extra_logs:
            mean_log.update(extra_logs)
        try:
            for param_group in self.optimizers().optimizer.param_groups:
                mean_log['lr'] = param_group['lr']
        except:
            pass

        if self.trainer.is_global_zero:
            logger.info(str(mean_log))
            if self.args.swanlab and SWANLAB_AVAILABLE:
                swanlab.log(mean_log)
            if save:
                path = os.path.join(
                    os.environ["MODEL_DIR"],
                    f"{prefix}_{self.trainer.current_epoch}.csv"
                )
                pd.DataFrame(log).to_csv(path)
        for key in list(log.keys()):
            if f"{prefix}_" in key:
                del self._log[key]

    def configure_optimizers(self):
        cls = torch.optim.AdamW if self.args.adamW else torch.optim.Adam
        optimizer = cls(
            # Use wrapper parameters so optional trainable modules outside
            # self.model (for example ptm_adapter) are not silently excluded.
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.args.lr,
        )

        warmup_epochs = getattr(self.args, 'warmup_epochs', 0)
        if warmup_epochs <= 0:
            return optimizer

        min_lr      = getattr(self.args, 'min_lr', 1e-6)
        total_epochs = getattr(self.args, 'epochs', 1000)

        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=min_lr / self.args.lr,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(total_epochs - warmup_epochs, 1),
            eta_min=min_lr,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )
        return {
            "optimizer":    optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1},
        }


class ProteinWrapper(Wrapper):
    def __init__(self, args):
        super().__init__(args)

        self.latent_dim = 21
        self.model = LatentModel(args, self.latent_dim)

        self.use_ptm_feat = hasattr(args, 'ptm_feat_path') and args.ptm_feat_path is not None
        if self.use_ptm_feat:
            input_feat_dim = int(getattr(args, 'ptm_feat_dim', 256))
            target_embed_dim = getattr(args, 'embed_dim', 384)
            self.ptm_adapter = nn.Sequential(
                nn.Linear(input_feat_dim, target_embed_dim),
                nn.SiLU(),
                nn.Linear(target_embed_dim, target_embed_dim)
            )
            nn.init.zeros_(self.ptm_adapter[-1].weight)
            nn.init.zeros_(self.ptm_adapter[-1].bias)

        self.transport = create_transport(args, args.path_type, args.prediction, None)
        self.transport_sampler = Sampler(self.transport)

        # PTM finetune: freeze backbone, LoRA-adapt attn/FFN, fully train the PTM
        # channel(s). Gated by --use_lora; base models (flag absent) are unaffected.
        # Applied before EMA creation so EMA tracks the LoRA/PTM parameters.
        if getattr(args, 'use_lora', False):
            from model.lora import apply_lora
            for p in self.model.parameters():
                p.requires_grad_(False)
            apply_lora(self.model, r=args.lora_r, alpha=args.lora_alpha,
                       dropout=args.lora_dropout)
            if getattr(self.model, 'use_ptm_channel', False):
                self.model.ptm_channel.weight.requires_grad_(True)
            if self.use_ptm_feat and hasattr(self, 'ptm_adapter'):
                for p in self.ptm_adapter.parameters():
                    p.requires_grad_(True)
            n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"[LoRA] trainable params: {n_train/1e6:.3f}M")

        if not hasattr(args, 'ema'):
            args.ema = False
        if args.ema:
            self.ema = ExponentialMovingAverage(model=self.model, decay=args.ema_decay)
            self.cached_weights = None

    def _prep_frame_batch(self, batch):
        rigids = Rigid(trans=batch['trans'], rots=Rotation(rot_mats=batch['rots']))
        B, T, L = rigids.shape

        offsets = get_offsets(rigids[:, 0:1], rigids)
        offsets[..., :4] *= torch.where(offsets[:, :, :, 0:1] < 0, -1, 1)

        frame_loss_mask = batch['mask'].unsqueeze(-1).expand(-1, -1, 7)
        torsion_loss_mask = batch['torsion_mask'].unsqueeze(-1).expand(-1, -1, -1, 2).reshape(B, L, 14)

        latents = torch.cat([offsets, batch['torsions'].view(B, T, L, 14)], -1)
        loss_mask = torch.cat([frame_loss_mask, torsion_loss_mask], -1)
        loss_mask = loss_mask.unsqueeze(1).expand(-1, T, -1, -1)

        cond_mask = torch.zeros(B, T, L, dtype=int, device=offsets.device)
        cond_mask[:, 0] = 1

        # Conditioning aatype: use the ablation override when present (geometry
        # below still uses the real batch['seqres']). For AA-base models this is
        # how the PTM signal (21/22/23 → aatype_to_emb) gets ablated.
        cond_aatype = batch.get('aatype_override', batch['seqres'])

        model_kwargs = {
            'start_frames': rigids[:, 0],
            'mask': batch['mask'].unsqueeze(1).expand(-1, T, -1),
            'aatype': cond_aatype,
            'x_cond': torch.where(cond_mask.unsqueeze(-1).bool(), latents, 0.0),
            'x_cond_mask': cond_mask,
        }

        if self.use_ptm_feat and hasattr(self, 'ptm_adapter') and 'ptm_feat' in batch:
            model_kwargs['ptm_emb'] = self.ptm_adapter(batch['ptm_feat'].to(self.device))

        if 'esm2_emb' in batch:
            model_kwargs['esm2_emb'] = batch['esm2_emb'].to(self.device)
        if 'ptm_ids_override' in batch:
            model_kwargs['ptm_ids'] = batch['ptm_ids_override'].to(self.device)

        return {
            'rigids': rigids,
            'latents': latents,
            'loss_mask': loss_mask,
            'model_kwargs': model_kwargs
        }

    def prep_batch(self, batch):
        return self._prep_frame_batch(batch)

    def general_step(self, batch, stage='train'):
        self.iter_step += 1
        self.stage = stage
        start1 = time.time()

        prep = self.prep_batch(batch)

        start = time.time()
        out_dict = self.transport.training_losses(
            model=self.model,
            x1=prep['latents'],
            mask=prep['loss_mask'],
            model_kwargs=prep['model_kwargs']
        )
        self.log('model_dur', time.time() - start)
        loss = out_dict['loss']
        self.log('loss', loss)

        # ── Ensemble-level distribution-matching loss (toggleable) ──────────
        if getattr(self.args, 'ensemble_loss', False) and stage == 'train':
            from trainer.ensemble_loss import (
                predict_x1_from_velocity, ensemble_distribution_loss)
            x1_hat = predict_x1_from_velocity(
                self.transport.path_sampler, out_dict['xt'], out_dict['t'], out_dict['pred'])
            B, T, L, C = x1_hat.shape
            res_mask = batch['mask'].bool() if 'mask' in batch else None
            d = ensemble_distribution_loss(
                x1_hat, out_dict['x1'], res_mask=res_mask,
                kind=getattr(self.args, 'ensemble_kind', 'energy'),
                feature=getattr(self.args, 'ensemble_feature', 'per_residue'),
                t=out_dict['t'], tau_min=getattr(self.args, 'ensemble_tau_min', 0.3))
            warmup = max(1, getattr(self.args, 'ensemble_warmup', 0))
            ramp = min(1.0, self.current_epoch / warmup)
            lam = getattr(self.args, 'ensemble_lambda', 0.1) * ramp
            loss = loss + lam * d
            self.log('ens_dist', d)
            self.log('ens_lambda', lam)

        self.log('time', out_dict['t'])
        self.log('dur', time.time() - self.last_log_time)
        if 'name' in batch:
            self.log('name', ','.join(batch['name']))
        self.log('general_step_dur', time.time() - start1)
        self.last_log_time = time.time()
        return loss.mean()

    def inference(self, batch):
        aatype_geom = map_ptm_to_parent(batch['seqres'])

        prep = self.prep_batch(batch)
        rigids = prep['rigids']
        B, T, L = rigids.shape

        zs = torch.randn(B, T, L, self.latent_dim, device=self.device)
        if getattr(self.args, 'sde', False):
            sample_fn = self.transport_sampler.sample_sde(
                num_steps=getattr(self.args, 'sde_steps', 100),
                churn=getattr(self.args, 'churn', 1.0))
        else:
            sample_fn = self.transport_sampler.sample_ode(sampling_method=self.args.sampling_method)
        samples = sample_fn(
            zs,
            partial(self.model.forward_inference, **prep['model_kwargs'])
        )[-1]

        offsets = samples[..., :7]
        torsions = samples[..., 7:21]

        frames = rigids[:, 0:1].compose(Rigid.from_tensor_7(offsets, normalize_quats=True))
        torsions = torsions.reshape(B, T, L, 7, 2)
        if not self.args.oracle:
            torsions = torsions / torch.linalg.norm(torsions, dim=-1, keepdims=True)

        atom14 = frames_torsions_to_atom14(
            frames,
            torsions.view(B, T, L, 7, 2),
            aatype_geom[:, None].expand(B, T, L)
        )
        aa_out = batch['seqres'][:, None].expand(B, T, L)
        return atom14, aa_out

    def validation_step_extra(self, batch, batch_idx):
        do_inference = batch_idx < self.args.inference_batches and (
            (self.current_epoch + 1) % self.args.designability_freq == 0 or self.args.validate
        ) and self.trainer.is_global_zero

        if do_inference:
            atom14, _ = self.inference(batch)
            prot_name = batch['name'][0]
            path = os.path.join(os.environ["MODEL_DIR"], f'epoch{self.current_epoch}_{prot_name}.pdb')
            atom14_to_pdb(atom14[0].cpu().numpy(), batch['seqres'][0].cpu().numpy(), path)

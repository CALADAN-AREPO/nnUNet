import torch
import torch.nn.functional as F

from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from daoct.losses.boundary_loss import DAOCTLoss


def _resize_target(tensor: torch.Tensor, target_spatial, mode='nearest'):
    if tensor.dim() == 4 and (tensor.shape[2] != target_spatial[0] or tensor.shape[3] != target_spatial[1]):
        tensor = F.interpolate(tensor.float(), size=target_spatial, mode=mode).long()
    return tensor


class DAOCTTrainerMixin:
    def _build_loss(self):
        return DAOCTLoss()

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        dist_maps = batch.get('dist_maps', None)
        if dist_maps is not None:
            dist_maps = dist_maps.to(self.device, non_blocking=True)

        output = self.network(data)

        if isinstance(output, (tuple, list)):
            weights = self._get_deep_supervision_weights(len(output))
            total_loss = 0.0
            for i, (oi, ti) in enumerate(zip(output, target)):
                if weights[i] == 0.0:
                    continue
                dm = dist_maps if i == 0 else None
                ti = _resize_target(ti, oi.shape[2:])
                l = self.loss(oi, ti, dist_maps=dm, epoch=self.current_epoch)
                total_loss += weights[i] * l
            l = total_loss
        else:
            l = self.loss(output, target, dist_maps=dist_maps, epoch=self.current_epoch)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}

    def _get_deep_supervision_weights(self, num_outputs=None):
        import numpy as np
        deep_supervision_scales = self._get_deep_supervision_scales()
        if num_outputs is not None and num_outputs < len(deep_supervision_scales):
            deep_supervision_scales = deep_supervision_scales[:num_outputs]
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        if self.is_ddp:
            weights[-1] = 1e-6
        else:
            weights[-1] = 0
        weights = weights / weights.sum()
        return weights

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        output = self.network(data)
        del data

        if isinstance(output, (tuple, list)):
            weights = self._get_deep_supervision_weights(len(output))
            l = 0.0
            for i, (oi, ti) in enumerate(zip(output, target)):
                if weights[i] == 0.0:
                    continue
                ti = _resize_target(ti, oi.shape[2:])
                l = l + weights[i] * self.loss(oi, ti)
        else:
            l = self.loss(output, target)

        output = output[0] if isinstance(output, (tuple, list)) else output
        target = target[0] if isinstance(target, (tuple, list)) else target
        if target.shape[2:] != output.shape[2:]:
            target = _resize_target(target, output.shape[2:])

        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float16)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}

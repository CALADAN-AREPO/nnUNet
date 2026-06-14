import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from daoct.architectures.encoders.retfound_vit_encoder import RETFoundViTUNet
from daoct.da.augmentations import CROSS_VENDOR_TRANSFORMS
from daoct.training.daoct_trainer_mixin import DAOCTTrainerMixin


class nnUNetTrainerDAOCT_RETFound(DAOCTTrainerMixin, nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 5e-5
        self.weight_decay = 1e-2
        self.num_epochs = 500

    @staticmethod
    def build_network_architecture(plans_manager, configuration_manager, num_input_channels,
                                   num_output_channels, enable_deep_supervision=True):
        return RETFoundViTUNet(
            num_classes=num_output_channels,
            deep_supervision=enable_deep_supervision,
        )

    def configure_optimizers(self):
        backbone = self.network.backbone
        projector = self.network.projector
        decoder = self.network.decoder
        reassemble = self.network.reassemble

        for i in range(6):
            if i < len(backbone.blocks):
                for p in backbone.blocks[i].parameters():
                    p.requires_grad_(False)

        ln_params = []
        for name, p in backbone.named_parameters():
            if "norm" in name.lower():
                ln_params.append(p)
                p.requires_grad = True

        trainable_backbone_params = []
        for name, p in backbone.named_parameters():
            if p.requires_grad and "norm" not in name.lower():
                trainable_backbone_params.append(p)

        param_groups = [
            {"params": projector.parameters(), "lr": 5e-4, "weight_decay": self.weight_decay},
            {"params": reassemble.parameters(), "lr": 5e-4, "weight_decay": self.weight_decay},
            {"params": decoder.parameters(), "lr": 5e-4, "weight_decay": self.weight_decay},
            {"params": ln_params, "lr": 1e-6, "weight_decay": 0},
            {"params": trainable_backbone_params, "lr": 5e-5, "weight_decay": self.weight_decay},
        ]

        optimizer = AdamW(param_groups, lr=self.initial_lr, weight_decay=self.weight_decay)
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs)
        return optimizer, lr_scheduler

    @staticmethod
    def get_training_transforms(
        patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
        use_mask_for_norm=None, is_cascaded=False,
        foreground_labels=None, regions=None, ignore_label=None,
    ):
        base = nnUNetTrainer.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm, is_cascaded, foreground_labels, regions, ignore_label,
        )
        for t in CROSS_VENDOR_TRANSFORMS:
            base.transforms.insert(-1, t)
        return base

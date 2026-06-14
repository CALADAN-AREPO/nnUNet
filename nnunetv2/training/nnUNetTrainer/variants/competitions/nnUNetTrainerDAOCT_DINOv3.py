import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from daoct.architectures.encoders.dinov3_convnext_encoder import DINOv3ConvNextUNet
from daoct.da.augmentations import CROSS_VENDOR_TRANSFORMS
from daoct.training.daoct_trainer_mixin import DAOCTTrainerMixin


class nnUNetTrainerDAOCT_DINOv3(DAOCTTrainerMixin, nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 1e-4
        self.weight_decay = 1e-2
        self.num_epochs = 500

    @staticmethod
    def build_network_architecture(plans_manager, configuration_manager, num_input_channels,
                                   num_output_channels, enable_deep_supervision=True):
        return DINOv3ConvNextUNet(
            num_classes=num_output_channels,
            variant="base",
            deep_supervision=enable_deep_supervision,
        )

    def configure_optimizers(self):
        backbone = self.network.backbone
        projector = self.network.projector
        channel_adapters = self.network.channel_adapters
        decoder = self.network.decoder

        param_groups = []
        param_groups.append({
            "params": projector.parameters(),
            "lr": 1e-3,
            "weight_decay": self.weight_decay,
        })
        param_groups.append({
            "params": channel_adapters.parameters(),
            "lr": 1e-3,
            "weight_decay": self.weight_decay,
        })
        decoder_params = []
        for name, param in decoder.named_parameters():
            if param.requires_grad:
                decoder_params.append(param)
        param_groups.append({
            "params": decoder_params,
            "lr": 1e-3,
            "weight_decay": self.weight_decay,
        })

        base_lr = self.initial_lr
        stage_decay = 0.75
        stages = ["stem", "stage1", "stage2", "stage3", "stage4"]
        stage_lr_mult = [stage_decay ** (len(stages) - i - 1) for i in range(len(stages))]

        for si, stage_name in enumerate(stages):
            if hasattr(backbone, stage_name):
                stage_module = getattr(backbone, stage_name)
                stage_params = []
                for name, param in stage_module.named_parameters():
                    if param.requires_grad:
                        stage_params.append(param)
                if stage_params:
                    param_groups.append({
                        "params": stage_params,
                        "lr": base_lr * stage_lr_mult[si],
                        "weight_decay": self.weight_decay,
                    })

        stage3 = getattr(backbone, "stage3", None)
        if stage3 is not None:
            stage3_blocks = list(stage3.children()) if hasattr(stage3, 'children') else []
            if len(stage3_blocks) > 18:
                for bi, block in enumerate(stage3_blocks):
                    if bi < 18:
                        for p in block.parameters():
                            p.requires_grad_(False)

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

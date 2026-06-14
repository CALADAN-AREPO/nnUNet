import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from daoct.architectures.encoders.medsam3_hiera_encoder import MedSAM3UNet
from daoct.da.augmentations import CROSS_VENDOR_TRANSFORMS
from daoct.training.daoct_trainer_mixin import DAOCTTrainerMixin


class nnUNetTrainerDAOCT_MedSAM3(DAOCTTrainerMixin, nnUNetTrainer):
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
        return MedSAM3UNet(
            num_classes=num_output_channels,
            deep_supervision=enable_deep_supervision,
            lora_r=4,
            lora_alpha=8,
        )

    def configure_optimizers(self):
        projector = self.network.projector
        decoder = self.network.decoder
        lora_params = [p for n, p in self.network.named_parameters() if 'lora_' in n]

        param_groups = [
            {"params": projector.parameters(), "lr": 1e-3, "weight_decay": self.weight_decay},
            {"params": decoder.parameters(), "lr": 5e-4, "weight_decay": self.weight_decay},
        ]
        if lora_params:
            param_groups.append(
                {"params": lora_params, "lr": 1e-4, "weight_decay": self.weight_decay}
            )

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

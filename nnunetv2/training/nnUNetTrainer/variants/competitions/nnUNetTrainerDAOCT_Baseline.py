from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from daoct.da.augmentations import CROSS_VENDOR_TRANSFORMS
from daoct.training.daoct_trainer_mixin import DAOCTTrainerMixin


class nnUNetTrainerDAOCT_Baseline(DAOCTTrainerMixin, nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, device=None):
        if device is None:
            import torch
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        super().__init__(plans, configuration, fold, dataset_json, device)

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

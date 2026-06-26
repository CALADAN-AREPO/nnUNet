from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
import torch

from nnunetv2.network_architecture.dual_scale_swin_mednext import DualScaleSwinMedNeXt


class nnUNetTrainerDualScaleSwin(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.enable_deep_supervision = False

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> torch.nn.Module:
        return DualScaleSwinMedNeXt(
            in_channels=num_input_channels,
            out_channels=num_output_channels,
            feature_size=48,
            mednext_kernel_size=5,
            mednext_exp_r=4,
            depths=(2, 2, 2, 2),
            num_heads=(3, 6, 12, 24),
            window_size=7,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            use_checkpoint=True,
            use_v2=True,
            scale2_factor=2.0,
        )

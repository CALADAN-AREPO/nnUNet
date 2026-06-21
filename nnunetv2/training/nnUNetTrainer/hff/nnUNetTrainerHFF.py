from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.nets.hff import HFF_UNet

import torch
from typing import Union, List, Tuple


class nnUNetTrainerHFF(nnUNetTrainer):
    """
    nnUNet trainer using the HFF_UNet (Hierarchical Feature Fusion) architecture.

    Designed for neonatal brain MRI segmentation from low-field CISO images.
    Uses attention-gated skip connections and multi-scale feature fusion in
    the decoder for improved small-structure segmentation.
    """
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
        self.num_epochs = 500

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> torch.nn.Module:
        return get_network_from_plans(
            "nnunetv2.nets.hff.HFF_UNet",
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision,
        )

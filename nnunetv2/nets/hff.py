from typing import Union, Type, List, Tuple, Optional

import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from torch.nn.modules.instancenorm import _InstanceNorm

from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0


class AttentionGate(nn.Module):
    """
    Attention gate for skip connections. Reweights the skip feature map
    using a gating signal from the coarser decoder level.
    """
    def __init__(
        self,
        F_g: int,
        F_l: int,
        F_int: int,
        conv_op: Type[_ConvNd],
        norm_op: Optional[Type[_InstanceNorm]] = None,
        norm_op_kwargs: Optional[dict] = None,
        nonlin: Optional[Type[nn.Module]] = None,
        nonlin_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.W_g = conv_op(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True)
        self.W_x = conv_op(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = nn.Sequential(
            conv_op(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )
        self.relu = nonlin(**nonlin_kwargs) if nonlin is not None else nn.ReLU()

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class FusionBlock(nn.Module):
    """
    Decoder block that fuses upsampled coarse features with attention-gated
    skip features and optionally a higher-level semantic guide.
    """
    def __init__(
        self,
        skip_channels: int,
        coarse_channels: int,
        output_channels: int,
        conv_op: Type[_ConvNd],
        kernel_size: int,
        norm_op: Optional[Type[_InstanceNorm]],
        norm_op_kwargs: Optional[dict],
        nonlin: Optional[Type[nn.Module]],
        nonlin_kwargs: Optional[dict],
        conv_bias: bool = False,
    ):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)

        attn_int = max(skip_channels // 2, 4)
        self.attn_gate = AttentionGate(
            F_g=coarse_channels,
            F_l=skip_channels,
            F_int=attn_int,
            conv_op=conv_op,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
        )

        conv_in_channels = skip_channels + coarse_channels
        self.conv = StackedConvBlocks(
            num_convs=2,
            conv_op=conv_op,
            input_channels=conv_in_channels,
            output_channels=output_channels,
            kernel_size=kernel_size,
            initial_stride=1,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=None,
            dropout_op_kwargs=None,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
        )

    def forward(self, coarse: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        up = self.upsample(coarse)
        attn_skip = self.attn_gate(up, skip)
        fused = torch.cat([up, attn_skip], dim=1)
        return self.conv(fused)


class HFF_UNet(nn.Module):
    """
    Hierarchical Feature Fusion U-Net for neonatal brain MRI segmentation.

    Uses a residual encoder (same as nnUNet's ResidualUNet) but replaces the
    standard decoder with attention-gated fusion blocks that weight skip
    connections by relevance to the upsampled coarse features.

    This is particularly beneficial for small structure segmentation in
    low-field MRI where subtle boundary details matter most.
    """
    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
        num_classes: int,
        n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[_InstanceNorm]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[nn.Module]] = None,
        nonlin_kwargs: dict = None,
        deep_supervision: bool = False,
        block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
        bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
        stem_channels: int = None,
    ):
        super().__init__()
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages

        self.deep_supervision = deep_supervision
        self.num_classes = num_classes

        self.encoder = ResidualEncoder(
            input_channels, n_stages, features_per_stage, conv_op, kernel_sizes,
            strides, n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs,
            dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, block,
            bottleneck_channels, return_skips=True, disable_default_stem=False,
            stem_channels=stem_channels,
        )

        decoder_features = features_per_stage[:0:-1]
        skip_features = features_per_stage[-2::-1]

        self.fusion_blocks = nn.ModuleList()
        for i in range(n_stages - 1):
            coarse_ch = decoder_features[i] if i > 0 else features_per_stage[-1]
            skip_ch = skip_features[i]
            out_ch = features_per_stage[n_stages - 2 - i]
            ks = kernel_sizes[min(n_stages - 1, len(kernel_sizes) - 1)]
            self.fusion_blocks.append(
                FusionBlock(
                    skip_channels=skip_ch,
                    coarse_channels=coarse_ch,
                    output_channels=out_ch,
                    conv_op=conv_op,
                    kernel_size=ks,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                    conv_bias=conv_bias,
                )
            )

        self.seg_head = conv_op(
            features_per_stage[0], num_classes,
            kernel_size=1, padding=0,
        )

        if deep_supervision:
            ds_in = features_per_stage[-2:0:-1]
            self.ds_heads = nn.ModuleList()
            for feat in ds_in:
                self.ds_heads.append(conv_op(feat, num_classes, kernel_size=1, padding=0))

        self.upsample_seg = nn.Upsample(
            scale_factor=2, mode="trilinear", align_corners=False,
        )

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        skips = self.encoder(x)
        # ResidualEncoder returns [stage0, stage1, ..., stageN-1] (shallowest to deepest)
        # skips[0] = stage0 (closest to input), skips[-1] = bottleneck (deepest)

        x = skips[-1]
        n_skips = len(skips)
        ds_outputs = []

        for i in range(len(self.fusion_blocks)):
            skip = skips[n_skips - 2 - i]
            x = self.fusion_blocks[i](x, skip)
            if self.deep_supervision and i < len(self.ds_heads):
                ds = self.ds_heads[i](x)
                for _ in range(len(self.fusion_blocks) - 1 - i):
                    ds = self.upsample_seg(ds)
                ds_outputs.append(ds)

        out = self.seg_head(x)

        if self.deep_supervision:
            return [out] + ds_outputs
        return out

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op)
        return self.encoder.compute_conv_feature_map_size(input_size) + sum(
            fb.conv.compute_conv_feature_map_size(input_size)
            for fb in self.fusion_blocks
            for _ in [0]
        )

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)
        init_last_bn_before_add_to_0(module)

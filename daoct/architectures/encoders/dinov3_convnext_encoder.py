import torch
import torch.nn as nn
from transformers import DINOv3ConvNextBackbone, DINOv3ConvNextConfig

from daoct.architectures.projectors.channel_projector import LearnedChannelProjector


DINOv3_CHANNELS = {
    "base": [128, 256, 512, 1024],
    "large": [192, 384, 768, 1536],
}
DINOv3_DECODER_CHANNELS = [32, 64, 128, 256]


class DINOv3ConvNextUNet(nn.Module):
    def __init__(self, num_classes=10, variant="base", deep_supervision=True):
        super().__init__()
        self.projector = LearnedChannelProjector()
        variant_key = variant if variant in DINOv3_CHANNELS else "base"
        backbone_channels = DINOv3_CHANNELS[variant_key]

        model_id = f"facebook/dinov3-convnext-{variant_key}-pretrain-lvd1689m"
        config = DINOv3ConvNextConfig.from_pretrained(
            model_id,
            out_indices=[1, 2, 3, 4],
            out_features=["stage1", "stage2", "stage3", "stage4"],
        )
        self.backbone = DINOv3ConvNextBackbone.from_pretrained(
            model_id, config=config,
        )

        self.channel_adapters = nn.ModuleList([
            nn.Conv2d(bc, dc, kernel_size=1)
            for bc, dc in zip(backbone_channels, DINOv3_DECODER_CHANNELS)
        ])

        from dynamic_network_architectures.architectures.unet import UNetDecoder

        conv_op = nn.Conv2d
        strides = [(4, 4), (2, 2), (2, 2), (2, 2)]
        kernel_sizes = [(1, 1)] * 4

        class EncoderProxy:
            def __init__(self, output_channels, conv_op, conv_bias, norm_op, norm_op_kwargs,
                         dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, strides, kernel_sizes):
                self.output_channels = output_channels
                self.conv_op = conv_op
                self.conv_bias = conv_bias
                self.norm_op = norm_op
                self.norm_op_kwargs = norm_op_kwargs
                self.dropout_op = dropout_op
                self.dropout_op_kwargs = dropout_op_kwargs
                self.nonlin = nonlin
                self.nonlin_kwargs = nonlin_kwargs
                self.strides = strides
                self.kernel_sizes = kernel_sizes

        self.encoder_proxy = EncoderProxy(
            output_channels=DINOv3_DECODER_CHANNELS,
            conv_op=conv_op,
            conv_bias=False,
            norm_op=nn.BatchNorm2d,
            norm_op_kwargs={},
            dropout_op=None,
            dropout_op_kwargs=None,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={"negative_slope": 1e-2, "inplace": True},
            strides=strides,
            kernel_sizes=kernel_sizes,
        )

        self.decoder = UNetDecoder(
            encoder=self.encoder_proxy,
            num_classes=num_classes,
            n_conv_per_stage=[2, 2, 2],
            deep_supervision=deep_supervision,
        )

    def forward(self, x):
        x3ch = self.projector(x)
        out = self.backbone(x3ch)
        feats = out.feature_maps
        skips = [adapter(f) for adapter, f in zip(self.channel_adapters, feats)]
        return self.decoder(skips)

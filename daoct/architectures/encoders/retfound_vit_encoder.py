import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from huggingface_hub import hf_hub_download

from daoct.architectures.projectors.channel_projector import LearnedChannelProjector


RETFOUND_BLOCK_IDX = [5, 11, 17, 23]
RETFOUND_HIDDEN_DIM = 1024
RETFOUND_DECODER_CHANNELS = [256, 256, 256, 256]
RETFOUND_REPO_ID = "YukunZhou/RETFound_mae_natureOCT"
RETFOUND_CKPT_FILENAME = "RETFound_mae_natureOCT.pth"


def _load_retfound_backbone():
    model = timm.create_model("vit_large_patch16_224", pretrained=False)
    ckpt_path = hf_hub_download(RETFOUND_REPO_ID, RETFOUND_CKPT_FILENAME)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model"]
    decoder_prefixes = ("decoder_", "mask_token")
    enc_sd = {k: v for k, v in sd.items() if not k.startswith(decoder_prefixes)}
    missing, unexpected = model.load_state_dict(enc_sd, strict=False)
    assert all("head" in k for k in missing), f"Unexpected missing keys: {missing}"
    model.head = nn.Identity()
    return model


class RETFoundViTUNet(nn.Module):
    def __init__(self, num_classes=10, deep_supervision=True):
        super().__init__()
        self.projector = LearnedChannelProjector()
        self.backbone = _load_retfound_backbone()

        self.reassemble = nn.ModuleList([
            nn.Conv2d(RETFOUND_HIDDEN_DIM, 256, kernel_size=1) for _ in range(4)
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
            output_channels=RETFOUND_DECODER_CHANNELS,
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
        b, _, h, w = x3ch.shape

        self.backbone.patch_embed.img_size = (h, w)
        x_patch = self.backbone.patch_embed(x3ch)
        grid_h, grid_w = h // 16, w // 16
        x_patch = x_patch.reshape(b, grid_h, grid_w, -1)
        self.backbone.dynamic_img_size = True
        x_patch = self.backbone._pos_embed(x_patch)

        block_outputs = []
        hx = x_patch
        for i, blk in enumerate(self.backbone.blocks):
            hx = blk(hx)
            if i in RETFOUND_BLOCK_IDX:
                block_outputs.append(hx[:, 1:])

        skips = []
        grid_h, grid_w = h // 16, w // 16
        for i, bo in enumerate(block_outputs):
            feat = bo.reshape(b, grid_h, grid_w, -1).permute(0, 3, 1, 2)
            feat = self.reassemble[i](feat)
            if i == 0:
                feat = F.interpolate(feat, scale_factor=4, mode="bilinear", align_corners=False)
            elif i == 1:
                feat = F.interpolate(feat, scale_factor=2, mode="bilinear", align_corners=False)
            elif i == 3:
                feat = F.max_pool2d(feat, kernel_size=2, stride=2)
            skips.append(feat)

        return self.decoder(skips)

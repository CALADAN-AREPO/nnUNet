import os
import torch
import torch.nn as nn

from daoct.architectures.projectors.channel_projector import LearnedChannelProjector


SAM3_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "sam3")

MEDSAM3_INPUT_SIZE = 336


class LoRALinear(nn.Linear):
    """Low-Rank Adaptation (LoRA) wrapping an existing nn.Linear.

    Inherits from nn.Linear and copies the original weights so that
    external code accessing `.weight` / `.bias` (e.g. fused kernels)
    continues to work.  Only the low-rank matrices A and B are
    trainable; the base weights are frozen.
    """
    def __init__(self, original: nn.Linear, r: int = 4, alpha: float = 8.0):
        super().__init__(original.in_features, original.out_features,
                         bias=original.bias is not None,
                         device=original.weight.device,
                         dtype=original.weight.dtype)
        with torch.no_grad():
            self.weight.copy_(original.weight)
            if self.bias is not None:
                self.bias.copy_(original.bias)

        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(
            torch.randn(original.in_features, r, device=original.weight.device) * 0.01
        )
        self.lora_B = nn.Parameter(
            torch.zeros(r, original.out_features, device=original.weight.device)
        )

    def forward(self, x):
        result = super().forward(x)
        result += (x @ self.lora_A @ self.lora_B) * self.scaling
        return result


def _inject_lora(module, r=4, alpha=8, target_names=('qkv', 'proj', 'fc1', 'fc2')):
    """Recursively replace target Linear layers with LoRALinear wrappers."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and any(t in name for t in target_names):
            setattr(module, name, LoRALinear(child, r=r, alpha=alpha))
        else:
            _inject_lora(child, r=r, alpha=alpha, target_names=target_names)


def _count_lora_params(module):
    return sum(p.numel() for name, p in module.named_parameters()
               if 'lora_' in name)


def _patch_sam3_trunk(trunk):
    input_size = trunk.blocks[0].attn.input_size
    for blk in trunk.blocks:
        if blk.window_size == 0:
            attn = blk.attn
            attn.input_size = input_size
            attn._setup_rope_freqs()


def _patch_addmm_act(vitdet_module):
    orig = vitdet_module.addmm_act

    def patched(activation, linear, mat1):
        bias = linear.bias.detach()
        w = linear.weight.detach()
        m = mat1.view(-1, mat1.shape[-1])
        if activation in (torch.nn.functional.relu, torch.nn.ReLU):
            y = torch.nn.functional.linear(m, w, bias)
            y = torch.nn.functional.relu(y)
            return y.view(mat1.shape[:-1] + (y.shape[-1],))
        if activation in (torch.nn.functional.gelu, torch.nn.GELU):
            y = torch.nn.functional.linear(m, w, bias)
            y = torch.nn.functional.gelu(y)
            return y.view(mat1.shape[:-1] + (y.shape[-1],))
        raise ValueError(f"Unexpected activation {activation}")

    vitdet_module.addmm_act = patched
    return orig


class MedSAM3UNet(nn.Module):
    def __init__(self, num_classes=10, deep_supervision=True, lora_r=4, lora_alpha=8):
        super().__init__()
        self.projector = LearnedChannelProjector()
        self.input_size = MEDSAM3_INPUT_SIZE

        import sys
        sys.path.insert(0, SAM3_DIR)

        import sam3.model.vitdet as vitdet_module
        _patch_addmm_act(vitdet_module)

        from sam3 import build_sam3_image_model
        sam3 = build_sam3_image_model(device="cpu", load_from_HF=True)
        _patch_sam3_trunk(sam3.backbone.vision_backbone.trunk)

        self.vision_backbone = sam3.backbone.vision_backbone
        self.vision_backbone.eval()
        for p in self.vision_backbone.parameters():
            p.requires_grad_(False)

        if lora_r > 0:
            trunk = self.vision_backbone.trunk
            _inject_lora(trunk, r=lora_r, alpha=lora_alpha)
            print(f"[MedSAM3] LoRA injected (r={lora_r}, alpha={lora_alpha}), "
                  f"trainable LoRA params: {_count_lora_params(trunk)}")

        from dynamic_network_architectures.architectures.unet import UNetDecoder

        conv_op = nn.Conv2d
        strides = [(4, 4), (2, 2), (2, 2), (2, 2)]
        kernel_sizes = [(1, 1)] * 4

        neck_channels = [256] * 4

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
            output_channels=neck_channels,
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
        x = nn.functional.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        x3ch = self.projector(x)
        sam3_out, sam3_pos, sam2_out, sam2_pos = self.vision_backbone(x3ch)
        skips = sam3_out
        return self.decoder(skips)

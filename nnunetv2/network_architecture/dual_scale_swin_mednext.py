import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Sequence

from monai.networks.blocks import UnetOutBlock
from monai.utils import ensure_tuple_rep

from nnunetv2.network_architecture.swin_unetr import SwinTransformer
from nnunet_mednext.network_architecture.mednextv1.blocks import MedNeXtBlock, MedNeXtUpBlock


class DualScaleSwinMedNeXt(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        feature_size: int = 48,
        mednext_kernel_size: int = 5,
        mednext_exp_r: int = 4,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        window_size: int = 7,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        use_checkpoint: bool = True,
        use_v2: bool = True,
        scale2_factor: float = 2.0,
    ):
        super().__init__()
        assert feature_size % 12 == 0, "feature_size must be divisible by 12"

        self.scale2_factor = scale2_factor
        self.normalize = True

        self.swinViT = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=ensure_tuple_rep(window_size, 2),
            patch_size=ensure_tuple_rep(2, 2),
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=2,
            downsample='merging',
            use_v2=use_v2,
        )

        C = feature_size
        level_channels = [C, 2*C, 4*C, 8*C, 16*C]

        self.fuse = nn.ModuleList([
            nn.Conv2d(c * 2, c, kernel_size=1, bias=False)
            for c in level_channels
        ])

        self.enc0_block = MedNeXtBlock(
            in_channels, C, exp_r=2, kernel_size=3,
            do_res=False, norm_type='group', dim='2d',
        )
        self.enc1_block = MedNeXtBlock(
            2*C, 2*C, exp_r=mednext_exp_r, kernel_size=mednext_kernel_size,
            do_res=True, norm_type='group', dim='2d',
        )
        self.enc2_block = MedNeXtBlock(
            4*C, 4*C, exp_r=mednext_exp_r, kernel_size=mednext_kernel_size,
            do_res=True, norm_type='group', dim='2d',
        )
        self.enc3_block = MedNeXtBlock(
            8*C, 8*C, exp_r=mednext_exp_r, kernel_size=mednext_kernel_size,
            do_res=True, norm_type='group', dim='2d',
        )
        self.bottleneck = MedNeXtBlock(
            16*C, 16*C, exp_r=mednext_exp_r, kernel_size=mednext_kernel_size,
            do_res=True, norm_type='group', dim='2d',
        )

        self.enc0_swin_block = MedNeXtBlock(
            C, C, exp_r=2, kernel_size=3,
            do_res=True, norm_type='group', dim='2d',
        )

        self.up5 = MedNeXtUpBlock(
            16*C, 8*C, exp_r=mednext_exp_r, kernel_size=mednext_kernel_size,
            do_res=False, norm_type='group', dim='2d',
        )
        self.dec5 = MedNeXtBlock(
            8*C, 8*C, exp_r=mednext_exp_r, kernel_size=mednext_kernel_size,
            do_res=True, norm_type='group', dim='2d',
        )

        self.up4 = MedNeXtUpBlock(
            8*C, 4*C, exp_r=mednext_exp_r, kernel_size=mednext_kernel_size,
            do_res=False, norm_type='group', dim='2d',
        )
        self.dec4 = MedNeXtBlock(
            4*C, 4*C, exp_r=mednext_exp_r, kernel_size=mednext_kernel_size,
            do_res=True, norm_type='group', dim='2d',
        )

        self.up3 = MedNeXtUpBlock(
            4*C, 2*C, exp_r=mednext_exp_r, kernel_size=mednext_kernel_size,
            do_res=False, norm_type='group', dim='2d',
        )
        self.dec3 = MedNeXtBlock(
            2*C, 2*C, exp_r=mednext_exp_r, kernel_size=mednext_kernel_size,
            do_res=True, norm_type='group', dim='2d',
        )

        self.up2 = MedNeXtUpBlock(
            2*C, C, exp_r=mednext_exp_r, kernel_size=mednext_kernel_size,
            do_res=False, norm_type='group', dim='2d',
        )
        self.dec2 = MedNeXtBlock(
            C, C, exp_r=mednext_exp_r, kernel_size=mednext_kernel_size,
            do_res=True, norm_type='group', dim='2d',
        )

        self.up1 = MedNeXtUpBlock(
            C, C, exp_r=mednext_exp_r, kernel_size=mednext_kernel_size,
            do_res=True, norm_type='group', dim='2d',
        )
        self.dec1 = MedNeXtBlock(
            C, C, exp_r=2, kernel_size=3,
            do_res=True, norm_type='group', dim='2d',
        )

        self.out = UnetOutBlock(spatial_dims=2, in_channels=C, out_channels=out_channels)

    def _encode(self, x):
        return self.swinViT(x, self.normalize)

    def forward(self, x):
        B, C_in, H, W = x.shape

        h_s1 = self._encode(x)

        x_s2 = F.interpolate(
            x, scale_factor=self.scale2_factor,
            mode='bicubic', align_corners=False,
        )
        h_s2 = self._encode(x_s2)

        fused = []
        for i, (f1, f2) in enumerate(zip(h_s1, h_s2)):
            f2_down = F.interpolate(
                f2, size=f1.shape[-2:],
                mode='area',
            )
            fused.append(self.fuse[i](torch.cat([f1, f2_down], dim=1)))

        enc0 = self.enc0_block(x)
        enc1 = self.enc1_block(fused[1])
        enc2 = self.enc2_block(fused[2])
        enc3 = self.enc3_block(fused[3])
        bot = self.bottleneck(fused[4])

        d = self.up5(bot) + enc3
        d = self.dec5(d)

        d = self.up4(d) + enc2
        d = self.dec4(d)

        d = self.up3(d) + enc1
        d = self.dec3(d)

        enc0_swin = self.enc0_swin_block(fused[0])
        d = self.up2(d) + enc0_swin
        d = self.dec2(d)

        d = self.up1(d) + enc0
        d = self.dec1(d)

        return self.out(d)

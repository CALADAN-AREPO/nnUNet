import torch
import torch.nn as nn
import torch.nn.functional as F


class DPTReassemble(nn.Module):
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        self.layers = nn.ModuleList()
        for i, in_ch in enumerate(in_channels_list):
            if i == 0:
                self.layers.append(nn.Sequential(
                    nn.Conv2d(in_ch, out_channels, kernel_size=1),
                    nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
                ))
            elif i == 1:
                self.layers.append(nn.Sequential(
                    nn.Conv2d(in_ch, out_channels, kernel_size=1),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                ))
            elif i == 2:
                self.layers.append(nn.Conv2d(in_ch, out_channels, kernel_size=1))
            elif i == 3:
                self.layers.append(nn.Sequential(
                    nn.Conv2d(in_ch, out_channels, kernel_size=1),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                ))
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * len(in_channels_list), out_channels, kernel_size=1),
            nn.GELU(),
        )

    def forward(self, features):
        reassembled = []
        for layer, feat in zip(self.layers, features):
            reassembled.append(layer(feat))
        fused = self.fusion(torch.cat(reassembled, dim=1))
        return fused

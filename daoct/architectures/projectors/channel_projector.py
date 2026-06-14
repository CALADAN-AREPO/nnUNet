import torch
import torch.nn as nn


class LearnedChannelProjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(1, 3, kernel_size=1, bias=True)
        nn.init.constant_(self.proj.weight, 1.0 / 3.0)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

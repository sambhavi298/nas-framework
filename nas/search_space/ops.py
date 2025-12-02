import torch
import torch.nn as nn

# Simple identity operation with channel/stride handling
class Identity(nn.Module):
    def __init__(self, C_in, C_out, stride):
        super().__init__()
        if stride != 1 or C_in != C_out:
            self.downsample = nn.Sequential(
                nn.Conv2d(C_in, C_out, 1, stride=stride, bias=False),
                nn.BatchNorm2d(C_out)
            )
        else:
            self.downsample = None

    def forward(self, x):
        if self.downsample is None:
            return x
        return self.downsample(x)


# Standard conv 3x3 block
class ReLUConvBN(nn.Sequential):
    def __init__(self, C_in, C_out, kernel, stride, padding):
        super().__init__(
            nn.ReLU(inplace=False),     # IMPORTANT FIX
            nn.Conv2d(C_in, C_out, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(C_out),
        )


# ⚡ FINAL OPTIMIZED OPS (FASTEST FOR CPU)
OPS = {
    "skip": lambda C_in, C_out, stride: Identity(C_in, C_out, stride),
    "conv_3x3": lambda C_in, C_out, stride: ReLUConvBN(C_in, C_out, 3, stride, 1),
}

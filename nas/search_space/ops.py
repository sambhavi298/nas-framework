import torch
import torch.nn as nn
import torch.nn.functional as F


class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(C_out)
        )

    def forward(self, x):
        return self.op(x)


class DepthwiseConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(C_in, C_in, kernel_size, stride=stride,
                      padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out)
        )

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):
    def __init__(self, C_in, C_out, stride):
        super().__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride

    def forward(self, x):
        if self.C_in == self.C_out:
            if self.stride == 1:
                return x.mul(0.0)
            return x[:, :, ::self.stride, ::self.stride].mul(0.0)
        else:
            shape = list(x.shape)
            shape[1] = self.C_out
            shape[2] //= self.stride
            shape[3] //= self.stride
            return torch.zeros(shape, dtype=x.dtype, device=x.device)


def PoolBN(pool_type, C_in, C_out, stride):
    if C_in == C_out:
        if pool_type == "avg":
            return nn.Sequential(
                nn.AvgPool2d(3, stride=stride, padding=1),
                nn.BatchNorm2d(C_out)
            )
        else:
            return nn.Sequential(
                nn.MaxPool2d(3, stride=stride, padding=1),
                nn.BatchNorm2d(C_out)
            )
    else:
        if pool_type == "avg":
            return nn.Sequential(
                nn.AvgPool2d(3, stride=stride, padding=1),
                nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(C_out)
            )
        else:
            return nn.Sequential(
                nn.MaxPool2d(3, stride=stride, padding=1),
                nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(C_out)
            )


OPS = {
    "conv_3x3": lambda C_in, C_out, stride: ReLUConvBN(C_in, C_out, 3, stride, 1),
    "conv_5x5": lambda C_in, C_out, stride: ReLUConvBN(C_in, C_out, 5, stride, 2),
    "dwconv_3x3": lambda C_in, C_out, stride: DepthwiseConv(C_in, C_out, 3, stride, 1),
    "skip": lambda C_in, C_out, stride: Identity() if (stride == 1 and C_in == C_out) else Zero(C_in, C_out, stride),
    "avg_pool": lambda C_in, C_out, stride: PoolBN("avg", C_in, C_out, stride),
    "max_pool": lambda C_in, C_out, stride: PoolBN("max", C_in, C_out, stride),
    "zero": lambda C_in, C_out, stride: Zero(C_in, C_out, stride),
}

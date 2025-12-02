import torch
import torch.nn as nn
from nas.search_space.ops import OPS

class FinalCell(nn.Module):
    def __init__(self, C_in, C_out, stride, op_name):
        super().__init__()
        self.op = OPS[op_name](C_in, C_out, stride)

        if stride != 1 or C_in != C_out:
            self.skip = nn.Conv2d(C_in, C_out, 1, stride, bias=False)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        return self.op(x) + self.skip(x)


class FinalModel(nn.Module):
    def __init__(self, genotype, init_channels=16, num_classes=10):
        super().__init__()

        self.genotype = genotype
        C = init_channels

        self.stem = nn.Sequential(
            nn.Conv2d(3, C, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(C)
        )

        layers = []
        C_in = C

        for i, op_name in enumerate(genotype):
            stride = 2 if i == len(genotype)//2 else 1
            C_out = C_in * 2 if stride == 2 else C_in

            layers.append(FinalCell(C_in, C_out, stride, op_name))
            C_in = C_out

        self.layers = nn.Sequential(*layers)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(C_in, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layers(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
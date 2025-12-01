import torch
import torch.nn as nn
import torch.nn.functional as F

from nas.search_space.ops import OPS


OP_NAMES = list(OPS.keys())


class MixedOp(nn.Module):
    def __init__(self, C_in, C_out, stride):
        super().__init__()
        self.ops = nn.ModuleList()
        for name in OP_NAMES:
            op = OPS[name](C_in, C_out, stride)
            self.ops.append(op)

    def forward(self, x, weights):
        out = 0
        for w, op in zip(weights, self.ops):
            out = out + w * op(x)
        return out


class MixedLayer(nn.Module):
    def __init__(self, C_in, C_out, stride):
        super().__init__()
        self.mixed_op = MixedOp(C_in, C_out, stride)

        self.adjust = None
        if stride != 1 or C_in != C_out:
            self.adjust = nn.Sequential(
                nn.Conv2d(C_in, C_out, 1, stride=stride, bias=False),
                nn.BatchNorm2d(C_out)
            )

    def forward(self, x, weights):
        out = self.mixed_op(x, weights)
        if self.adjust is not None:
            x = self.adjust(x)
        return out + x


class SuperNet(nn.Module):
    def __init__(self, init_channels=16, num_layers=6, num_classes=10):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, init_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(init_channels)
        )

        C = init_channels
        layers = []
        for i in range(num_layers):
            stride = 2 if i == num_layers // 2 else 1
            next_C = C * 2 if stride == 2 else C
            layers.append(MixedLayer(C, next_C, stride))
            C = next_C

        self.layers = nn.ModuleList(layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(C, num_classes)

        self.alphas = nn.Parameter(1e-3 * torch.randn(num_layers, len(OP_NAMES)))

    def forward(self, x):
        weights = F.softmax(self.alphas, dim=1)

        x = self.stem(x)
        for i, layer in enumerate(self.layers):
            x = layer(x, weights[i])

        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)

    def genotype(self):
        idx = torch.argmax(self.alphas, dim=1).cpu().tolist()
        return [OP_NAMES[i] for i in idx]

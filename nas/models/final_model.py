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
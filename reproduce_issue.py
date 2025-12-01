import torch
from nas.search_space.ops import OPS
import sys

def test_ops_shapes():
    C_in = 16
    C_out = 32
    stride = 2
    x = torch.randn(1, C_in, 32, 32)
    
    with open("output_debug.txt", "w", encoding="utf-8") as f:
        f.write(f"Input shape: {x.shape}\n")
        f.write(f"Testing ops with stride={stride}, C_in={C_in}, C_out={C_out}\n")
        
        for name, op_factory in OPS.items():
            try:
                op = op_factory(C_in, C_out, stride)
                out = op(x)
                f.write(f"Op: {name}, Output shape: {out.shape}\n")
            except Exception as e:
                f.write(f"Op: {name}, Error: {e}\n")

if __name__ == "__main__":
    test_ops_shapes()

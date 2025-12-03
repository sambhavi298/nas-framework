# NAS Framework — Hardware-Aware Neural Architecture Search

This repository implements a clean, modular, research-grade Neural Architecture Search (NAS) framework designed for accuracy, latency, and hardware-aware optimization on edge devices and GPUs.

Current progress:

- Complete environment + project scaffolding  
- CIFAR-10 baseline (MobileNetV2)  
- Fully functional **search space (OPS)**  
- Final stable **SuperNet** (with correct shape-safe MixedOps)  
- Clean GitHub structure + synced commits  
- Ready for DARTS architecture search  

---

## Project Structure

```text
nas-framework/
│
├── configs/                 # (future) YAML configs
│
├── data/                    # datasets (ignored by git)
│
├── nas/
│   ├── models/
│   │   └── supernet.py      # final stable SuperNet implementation
│   │
│   ├── search_space/
│   │   └── ops.py           # final stable OPS (search space)
│   │
│   ├── trainers/            # (future) DARTS trainer
│   │
│   └── __init__.py
│
├── scripts/
│   └── train_baseline.py    # CIFAR-10 baseline
│
├── utils/                   # helpers (future)
│
└── README.md
```

---

##  Search Space (OPS)

The search space defines all candidate operations the NAS algorithm can choose from.

Includes:

- 3×3 conv  
- 5×5 conv  
- depthwise conv  
- skip connection (only when shape matches)  
- avg pooling  
- max pooling  
- zero op  
- all ops return **same output shape**  
- pooling + zero ops include **channel projection**  
- no dimension mismatch during MixedOp summation  

This gives a clean, DARTS-compatible operation set.

---

## SuperNet

The SuperNet implements a one-shot, over-parameterized model with:

- stem → MixedLayers → global pooling → classifier  
- mid-network downsampling  
- MixedOp with correct output sizes  
- learnable architecture weights (α)  
- `genotype()` for extracting the final architecture  

Verified with forward pass on CIFAR-sized inputs.

---

## Baseline (CIFAR-10)

Run:

```bash
python scripts/train_baseline.py
```

Confirms:

- torchvision dataset loads correctly  
- training loop works end-to-end  
- environment stable  

---

## Progress Achieved

- Conda environment created + dependencies fixed  
- PyTorch + Torchvision working  
- NumPy 2.x issue resolved  
- CIFAR-10 baseline implemented  
- Added final OPS (shape-safe, channel-projected)  
- Added final SuperNet (working MixedOps + residuals)  
- Project structure standardized  
- Clean GitHub history with .gitignore  
- Verified SuperNet output & genotype extraction  

Framework is now ready for implementing the **DARTS trainer**.

---

## Next Step

Upcoming module:

**DARTS Trainer (bi-level optimization for architecture search)**  
→ alternates weight updates (W) and architecture updates (α)  
→ extracts final architecture from optimized α  

---


## Final architecture discovered by NAS

['conv_3x3', 'skip', 'conv_3x3', 'conv_3x3', 'conv_3x3', 'conv_3x3']

Search completed in ~11 hours on CPU.
Peak validation accuracy during search: ~79.38%


## Final Training Results

- Train Accuracy: ~69%
- Test Accuracy: ~67.45%
- Training Time: ~45 minutes on CPU
- Search Time: ~11 hours on CPU


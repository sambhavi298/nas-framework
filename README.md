# NAS Framework — Hardware-Aware Neural Architecture Search

This repository contains the initial setup for a Neural Architecture Search (NAS) framework designed to discover efficient deep learning models optimized for **accuracy**, **latency**, and **memory** on target hardware (edge devices, CPUs, GPUs).

The project currently includes:

- Working baseline training pipeline (MobileNetV2 + CIFAR-10)
- Clean and scalable project structure
- Fully stable conda environment
- Repository prepared for NAS extensions (search space, supernet, hardware-aware profiling)

This forms the foundation for building a full NAS system.

---

## 📁 Project Structure

```text
nas-framework/
│
├── configs/               # Configuration files (search, model, hardware)
├── data/                  # Local datasets (ignored by Git)
├── datasets/              # Custom dataset loaders
├── hardware/
│   ├── profilers/         # Latency & resource profilers
│   └── cost_models/       # Learned latency cost models
├── models/                # Supernet + architecture models (WIP)
├── nas/
│   ├── algorithms/        # NAS algorithms (DARTS / ENAS / RL)
│   ├── evaluators/        # Accuracy + latency evaluation modules
│   ├── search_space/      # Search operations / cells
│   └── trainers/          # Supernet + architecture training loops
├── scripts/
│   └── train_baseline.py  # MobileNetV2 baseline training script
├── utils/                 # Helper utilities
└── README.md
---
Environment Setup

Create the environment:

conda create -n nas python=3.10 -y
conda activate nas


Install required packages:

pip install torch torchvision "numpy<2" pandas pyyaml tqdm
---

Baseline Model (MobileNetV2 + CIFAR-10)

Run training:

python scripts/train_baseline.py


Expected output:

epoch: 0  val_acc: 0.37
epoch: 1  val_acc: 0.48


This confirms:

CIFAR-10 loads correctly

Torch + Torchvision working

Training loop functioning

Environment stable
---
Next Steps (NAS Development)

1️⃣ Define search space operations
2️⃣ Implement the supernet (weight-sharing)
3️⃣ Add differentiable architecture parameters (α)
4️⃣ Build hardware latency profiler
5️⃣ Implement multi-objective NAS (accuracy + latency)
---
Notes

data/ is intentionally ignored to prevent large git commits

Future components will be added step-by-step

Baseline is only the starting point

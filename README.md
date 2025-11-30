# NAS Framework — Hardware-Aware Neural Architecture Search

This repository contains the initial setup for a Neural Architecture Search (NAS) framework designed to discover efficient deep learning models optimized for **accuracy**, **latency**, and **memory** on target hardware (edge devices, CPUs, GPUs).

The project currently includes:

- Fully working **baseline training pipeline**
- Stable **conda environment**
- Organized **project directory structure**
- CIFAR-10 training using **MobileNetV2**
- Git-clean repository prepared for NAS development

This serves as a foundation for implementing supernets, search algorithms, and hardware-aware profiling.

---

## Project Structure

nas-framework/
│
├── configs/ # Configuration files (search, model, hardware)
├── data/ # Local datasets (ignored by Git)
├── datasets/ # Custom dataset loaders
├── hardware/
│ ├── profilers/ # Latency & resource profilers
│ └── cost_models/ # Learned latency cost models
├── models/ # Supernet + architecture models (WIP)
├── nas/
│ ├── algorithms/ # NAS algorithms (DARTS / ENAS / RL)
│ ├── evaluators/ # Accuracy + latency evaluation modules
│ ├── search_space/ # Search operations / cells
│ └── trainers/ # Supernet + architecture training loops
├── scripts/
│ └── train_baseline.py # MobileNetV2 baseline training script
├── utils/ # Helper utilities
└── README.md


---

## 🛠 Environment Setup

Create the environment:

```bash
conda create -n nas python=3.10 -y
conda activate nas

Install required packages:
pip install torch torchvision "numpy<2" pandas pyyaml tqdm

## Baseline Model (MobileNetV2 + CIFAR-10)

A simple baseline for testing the pipeline.

Run training:
python scripts/train_baseline.py

Expected output:
epoch: 0  val_acc: 0.37
epoch: 1  val_acc: 0.48

This confirms:

Torch + Torchvision are working

CIFAR-10 loads correctly

Model trains end-to-end

Environment setup is stable

## Next Steps (NAS Development)
1. Search Space

Operations such as:

3×3 conv

5×5 conv

depthwise conv

skip connection

pooling ops

2. Supernet

A weight-sharing network containing all candidate operations.

3. Architecture Parameters (α)

DARTS-style differentiable architecture search.

4. Hardware Latency Profiling

Measure latency using:

PyTorch CPU/GPU inference timers

ONNX Runtime

Cost models

5. Multi-Objective Search

Optimize for:

accuracy

latency

memory footprint

## Notes

The data/ folder is ignored using .gitignore.

Large files (datasets, checkpoints) should not be pushed to GitHub.

The current baseline is only the starting point — NAS components will be added incrementally.

---

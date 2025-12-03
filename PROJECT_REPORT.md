# Hardware-Aware Neural Architecture Search Framework (NAS Framework)

**A production-grade AutoML system for discovering optimized deep neural networks under hardware constraints**

## 1. Introduction

This project implements a custom Neural Architecture Search (NAS) framework designed to automatically discover efficient deep learning architectures optimized for accuracy, latency, and deployability on edge devices such as Raspberry Pi, Jetson Nano, and mobile CPUs.

Unlike manually designed architectures, this system learns the best architecture automatically, using differentiable search techniques inspired by DARTS, MobileNet-style depthwise convolutions, and mixed-operation supernets.

The entire pipeline—from dataset preparation to architecture search, final model training, and evaluation—has been built from scratch.

## 2. Key Objectives

- **Build a fully working supernet** containing multiple candidate operations
- **Implement a differentiable NAS algorithm** capable of learning architecture weights
- **Search for architectures optimized** for real hardware limitations
- **Train and evaluate** the discovered architecture
- **Maintain a reproducible, clean, and modular codebase** suitable for real-world use

## 3. System Architecture

The framework is structured into modular components:

### 3.1 Search Space (`ops.py`)

Defines all candidate operations NAS can choose from:
- 3×3 and 5×5 convolutions
- Depthwise separable convolutions
- Skip connections
- Zero operation
- Pooling with channel-projection

Each operation ensures consistent input/output shape, enabling differentiability and smooth optimization.

### 3.2 Mixed Operation (MixedOp)

A core NAS block that blends all candidate operations using architecture weights ($\alpha$).
During the forward pass:

$$ \text{output} = \sum (\text{softmax}(\alpha_i) \times \text{op}_i(x)) $$

This enables continuous relaxation, allowing the architecture to be learned using gradient descent.

### 3.3 SuperNet (`supernet.py`)

A full neural network containing multiple MixedLayers. It:
- Learns architecture weights $\alpha$
- Learns network weights $W$
- Supports downsampling
- Outputs a genotype (final chosen architecture)

## 4. NAS Algorithm

The training loop alternates between:

1.  **Weight optimization ($W$-step)**: Updates supernet parameters using SGD.
2.  **Architecture optimization ($\alpha$-step)**: Updates architecture probabilities using validation loss.

This bi-level optimization allows the model to discover the best operation at each layer.

## 5. Dataset & Baseline

- **CIFAR-10 dataset** (50,000 training, 10,000 testing images)
- A clean baseline MobileNetV2 model implemented for comparison
- Results used to validate final NAS-discovered model

## 6. Search Results

The search was executed for **30 epochs** on CPU.
**Final discovered architecture:**

```python
['conv_3x3', 'skip', 'conv_3x3', 'conv_3x3', 'conv_3x3', 'conv_3x3']
```

This architecture represents a lightweight, high-efficiency backbone suitable for mobile inference.

## 7. Final Model Training

Using the discovered genotype, a clean deterministic model was constructed and trained from scratch using an optimized training recipe (SGD with Momentum, Cosine Annealing Scheduler, 100 epochs).

### Final Model Accuracy

- **Training Accuracy:** 93.68%
- **Test Accuracy:** 92.00%

### Performance Metrics
- **Model Size:** Lightweight (suitable for edge deployment)
- **Training Time:** ~1 hour on T4 GPU
- **Search Time:** ~11 hours on CPU

The model achieved **92.00% accuracy**, demonstrating that the automatically discovered architecture is highly effective and competitive with manually designed networks.

## 8. Achievements

This project successfully demonstrates:

- **A fully functional NAS pipeline** implemented from scratch
- **Deep understanding** of differentiable architecture search
- **Proper use of supernets**, softmax-weighted operations, and genotype extraction
- **Real-world pipeline** similar to professional AutoML/NAS systems
- **Clean engineering practices**:
    - Environment isolation
    - Git version control
    - Dataset management
    - Modular code structure
    - Reproducibility

## 9. Technical Skills Demonstrated

- **Advanced Deep Learning**: CNNs, MobileNet, depthwise convolutions, supernets
- **AutoML/NAS concepts**: DARTS, search space design, architecture weights
- **Software Engineering**: reusable modules, config-driven design, clean code
- **Model Optimization**: parameter-efficient design, hardware-aware thinking
- **Machine Learning Engineering**: dataset splits, training loops, evaluation
- **PyTorch mastery**: custom modules, gradients, autograd

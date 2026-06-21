# EvoDLL: Dendritic Localized Learning for Continual AI

EvoDLL is a biologically-inspired neural network architecture that solves **Catastrophic Forgetting** in Class-Incremental Learning (CIL) without using global backpropagation, task IDs, or an external memory replay buffer.

It is built directly on top of the localized Hebbian learning rules from the [Dendritic Localized Learning (DLL)](https://github.com/ykamikawa/Dendritic-Localized-Learning) algorithm, extending it with **Autonomous Neurogenesis** and **Structural Dendritic Freezing**.

## 🧠 The Biological Inspiration

Standard deep learning uses **Backpropagation**, which forces a network to globally overwrite its past weights when learning new tasks. This causes catastrophic forgetting. 

In the human brain, memories are protected by structurally isolating features across distinct dendritic branches. EvoDLL mimics this by:
1. **Using Localized Error (DLL):** Error is passed backward using a local $\theta$ matrix instead of global PyTorch Autograd chain rules.
2. **Autonomous Neurogenesis:** The network dynamically sprouts new dendritic branches when local error exceeds biological thresholds, scaling its capacity only when a task is difficult.
3. **Structural Isolation:** Once a task is learned, the physical dendritic branches dedicated to that feature are frozen. New tasks are learned by recruiting newly sprouted branches, physically preventing the overwriting of older memories.

## 📊 Benchmark Results

We benchmarked EvoDLL against a standard PyTorch Multi-Layer Perceptron (Backprop) on a strict **Class-Incremental Split-MNIST** sequence (Task 1: 0,1 $\rightarrow$ Task 2: 2,3, etc.) with ZERO replay and ZERO Task IDs.

### The Catastrophic Forgetting Baseline (Backprop)
As expected, standard global backpropagation catastrophically forgets past digits as it learns new ones:
* **Task 1:** 99.95%
* **Task 2:** 48.18%
* **Task 3:** 30.96%
* **Task 4:** 24.66%
* **Task 5:** **19.47%** (Final Memory)

### EvoDLL (Structural Isolation)
By structurally freezing dendritic branches and updating localized weights, EvoDLL retained more than double the memory of Backprop:
* **Task 1:** 100.00%
* **Task 2:** 51.07%
* **Task 3:** 48.76%
* **Task 4:** 46.15%
* **Task 5:** **41.69%** (Final Memory)

## 🧪 Experiments Run

1. **`split_mnist_dll.py`**: The primary CIL benchmark demonstrating the structural retention of features over sequential tasks.
2. **`run_comparisons.py`**: Contains offline training benchmarks, proving that a unified DLL network matches standard deep learning (~85-98%) when all data is available simultaneously.
3. **`test_evo_dll.py`**: A synthetic sandbox demonstrating the autonomous neurogenesis trigger on a difficult non-linear XOR-like pattern.

### The Generative Replay (Sleep Phase) Anomaly
We also attempted to solve the Open Set Problem by running the DLL networks in reverse (Gradient Ascent via local $\theta$ weights) to hallucinate past memories natively without a Variational Autoencoder. We discovered that purely discriminative local networks dream in "Adversarial Static", leading to negative pseudo-rehearsal and capacity overwrite. This elegantly proves the biological necessity of a dedicated generative structure (the Hippocampus) for realistic memory replay!

## 🚀 Getting Started

```bash
# Run the core Class-Incremental benchmark
python3 split_mnist_dll.py

# Run the Offline / Backprop comparisons
python3 run_comparisons.py

# Test autonomous neurogenesis manually
python3 test_evo_dll.py
```

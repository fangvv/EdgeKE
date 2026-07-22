## EdgeKE

[![GitHub](https://img.shields.io/badge/Project_Homepage-181717?logo=github)](https://github.com/fangvv/EdgeKE) — [https://github.com/fangvv/EdgeKE](https://github.com/fangvv/EdgeKE)

This is the source code for our paper: **EdgeKE: An On-Demand Deep Learning IoT System for Cognitive Big Data on Industrial Edge Devices**. A brief introduction of this work is as follows:

> Motivated by the prospects of 5G communications and industrial Internet of Things (IoT), recent years have seen the rise of a new computing paradigm, edge computing, which shifts data analytics to network edges that are at the proximity of big data sources. Although Deep Neural Networks (DNNs) have been extensively used in many platforms and scenarios, they are usually both compute and memory intensive, thus difficult to be deployed on resource-limited edge devices and in performance-demanding edge applications. Hence, there is an urgent need for techniques that enable DNN models to fit into edge devices, while ensuring acceptable execution costs and inference accuracy. This paper proposes an on-demand DNN model inference system for industrial edge devices, called Knowledge distillation and Early-exit on Edge (EdgeKE). It focuses on two design knobs: (1) DNN compression based on knowledge distillation, which trains the compact edge models under the supervision of large complex models for improving accuracy and speed; (2) DNN acceleration based on early-exit, which provides flexible choices for satisfying distinct latency or accuracy requirements from edge applications. By extensive evaluations on the CIFAR100 dataset and across three state-of-art edge devices, experimental results demonstrate that EdgeKE significantly outperforms the baseline models in terms of inference latency and memory footprint, while maintaining competitive classification accuracy. Furthermore, EdgeKE is verified to be efficiently adaptive to the application requirements on inference performance. The accuracy loss is within 4.84% under various latency constraints, and the speedup ratio is up to 3.30$\times$ under various accuracy requirements.

> 得益于5G通信与工业物联网（IIoT）的发展前景，近年来兴起了一种新型计算范式——边缘计算，其将数据分析任务转移至更接近大数据源的网络边缘。尽管深度神经网络（DNN）已被广泛应用于众多平台和场景，但其通常对计算和内存资源要求较高，难以部署在资源有限的边缘设备及性能要求严格的边缘应用中。因此，亟需一种能使DNN模型适配边缘设备，同时确保可接受的执行成本与推理精度的技术。本文提出了一种面向工业边缘设备的按需DNN模型推理系统EdgeKE（知识蒸馏与提前退出边缘系统），其核心聚焦两个设计维度：（1）基于知识蒸馏的DNN压缩技术，通过大型复杂模型监督训练紧凑型边缘模型，以提升精度与速度；（2）基于提前退出的DNN加速机制，为满足边缘应用不同延迟或精度需求提供灵活选择。通过在CIFAR100数据集及三种主流边缘设备上的广泛评估，实验结果表明：EdgeKE在推理延迟和内存占用方面显著优于基线模型，同时保持具有竞争力的分类精度。此外，EdgeKE被验证能有效自适应推理性能的应用需求——在不同延迟约束下精度损失控制在4.84%以内，在不同精度要求下加速比最高可达3.30倍。

This work has been published by IEEE Transactions on Industrial Informatics. Click [here](https://ieeexplore.ieee.org/document/9294146/) for our paper.

## Required software

- PyTorch
- torchvision
- NumPy
- SciPy

## Project Structure

```
EdgeKE/
├── datasets/
│   └── get_cifar_100.py                     # CIFAR-100 dataset loader
├── functions/
│   ├── my_functions.py                      # Core utilities: training, evaluation, knowledge distillation
│   └── branch_functions.py                  # BranchyNet: early-exit inference, voting, training strategies
├── model/
│   └── ResNet/
│       ├── ResNet.py                        # ResNet model definitions (ResNet18/34/50)
│       ├── get_ResNet_model.py              # Factory functions for student branch models & teacher model
│       └── get_inference_model.py           # Inference-time early-exit model architectures
├── main/
│   └── ResNet/
│       ├── ResNet_KD.py                     # Entry: standalone training, teacher training, KD training, evaluation
│       └── ResNet_KD_Branch.py              # Entry: BranchyNet training with knowledge distillation
├── test.py                                  # Model summary & quick test utilities
├── new_test.py                              # BranchyNet inference evaluation
├── temp.py                                  # Segmented inference timing evaluation
└── README.md
```

## Core Modules

### Dataset Loader (`datasets/get_cifar_100.py`)

Provides the `get_data()` function for loading the CIFAR-100 dataset with standard data augmentation (random cropping, horizontal flipping) and normalization.

**Key components:**
- `get_data(train_batch_size, test_batch_size)` — Returns `(train_loader, test_loader)` for CIFAR-100 with transforms applied.

### Core Utility Functions (`functions/my_functions.py`)

Implements the fundamental training, evaluation, and knowledge distillation pipelines. This is the core module that ties together all training workflows.

**Key components:**
- `copy_layer_param(old_model, new_model, layer_count)` — Copies parameters from one model to another up to a specified layer index, used for parameter sharing in BranchyNet.
- `adjust_learning_rate(optimizer, epoch)` / `fine_tune_adjust_learning_rate(optimizer, epoch)` — Step-decay learning rate schedules for standard training and fine-tuning.
- `get_Wasserstein_distance(tensor_A, tensor_B)` / `get_KL_divergence(tensor_A, tensor_B)` / `get_JS_divergence(tensor_A, tensor_B)` — Distance metrics (Wasserstein, KL, JS divergence) between softmax outputs, used as knowledge distillation loss.
- `get_project_dir()` — Returns the project root directory for resolving relative paths.
- `Eval_model(model, test_loader)` — Evaluates a single model, returning prediction accuracy, loss, total time, and per-sample inference time.
- `Train_model(model, Temperature, Epoch, DataSet, ...)` — Standard training loop with SGD optimizer, step-decay LR, and evaluation logging. Supports CIFAR-100 dataset.
- `Knowledge_distillation(Student_model, Teacher_model, Epoch, Temperature, beta, Distance_type, ...)` — Knowledge distillation training loop. Supports KL divergence, JS divergence, and Wasserstein distance as the distillation loss. Optionally freezes early layers of the student model via `copy_range`.

### BranchyNet Functions (`functions/branch_functions.py`)

Implements the **early-exit (BranchyNet)** mechanism, the second core design knob of EdgeKE. Includes training strategies (synchronous/asynchronous) with optional knowledge distillation, plus an inference-time early-exit evaluation pipeline with threshold-based exiting and weighted voting.

**Key components:**
- `get_threshold(outputs)` — Computes the confidence threshold (max softmax probability) from model outputs.
- `get_voting_softmax(output_list, voting_weight)` — Aggregates outputs from multiple exit points via weighted softmax voting.
- `get_Exit_Threshold(branch_model, test_loader, proportion)` — Calibrates per-exit confidence thresholds based on a target exit proportion on the test set.
- `get_loss(outputs, labels)` — Computes weighted joint loss across all exit points (weights: [0.2, 0.2, 0.6]).
- `Eval_BranchyNet(branch_model, exit_threshold, voting_weight, test_loader)` — Evaluates the full BranchyNet pipeline with early-exit: samples flow through branch models sequentially; if confidence exceeds the threshold, the sample exits early; otherwise it proceeds to deeper branches. Reports per-exit accuracy, exit percentage, and inference time. Samples that reach the final exit without meeting the threshold undergo weighted voting.
- `Train_BranchyNet_Synchronization(branch_model, Epoch, copy_range, ...)` — **Synchronous training**: all branch models are jointly optimized with shared loss, and parameters are copied from the main model to earlier branches after each batch.
- `Train_BranchyNet_Asynchronous(branch_model, main_Epoch, branch_Epoch, copy_range, ...)` — **Asynchronous training**: the main model is trained first, then earlier branches are trained while freezing shared parameters from deeper layers.
- `Train_BranchyNet_Asynchronous_Back(branch_model, Epoch_list, copy_range, ...)` — Asynchronous training variant that trains from the shallowest branch to the deepest.
- `Train_BranchyNet_Asynchronous_KD(branch_model, teacher_model, Epoch_list, copy_range, Distance_type, Temperature, beta, ...)` — **Asynchronous training with Knowledge Distillation**: combines both EdgeKE design knobs — trains branch models asynchronously with an additional distillation loss from the teacher model.
- `Train_BranchyNet_Asynchronous_KD_Back(...)` — Reverse-order variant of the above.

### ResNet Model Definitions (`model/ResNet/ResNet.py`)

Standard ResNet architectures implemented with `BasicBlock` and `Bottleneck` building blocks.

**Key components:**
- `BasicBlock` — Standard 2-layer residual block (3x3 conv → 3x3 conv).
- `Bottleneck` — 3-layer bottleneck block (1x1 → 3x3 → 1x1).
- `ResNet` — Generic ResNet backbone (`conv1` → `layer1-4` → `avgpool` → `linear`).
- `ResNet18(num_classes)` / `ResNet34(num_classes)` / `ResNet50()` — Factory functions.

### Model Factory (`model/ResNet/get_ResNet_model.py`)

Constructs the student (branch) models and teacher model for the EdgeKE system.

**Key components:**
- `get_model(num_classes=100)` — Returns `(main_model, branch_model_list)`, where the branch list contains three ResNet-based classifiers of increasing depth: a 3-section model (shallow), a 4-section model (medium), and ResNet18 (deep/main). These correspond to the three early-exit points in BranchyNet.
- `get_teacher_model(num_classes=100)` — Returns a ResNet34 as the teacher model for knowledge distillation.
- `ResNet_0` / `ResNet_1` / `ResNet_2` — Truncated ResNet variants used as early-exit branch classifiers.

### Inference Model Factory (`model/ResNet/get_inference_model.py`)

Defines the inference-time model architectures with segmented execution for fine-grained latency measurement. Instead of running complete branch models end-to-end, the inference pipeline splits each branch into segments, allowing per-segment timing.

**Key components:**
- `get_model(num_classes=100)` — Returns a list of 5 segmented models (`model_0` through `model_4`) that partition the computation graph. For example, `model_0` runs the first segment shared by all branches, `model_1` runs the first exit classifier, `model_2` runs the second segment, etc. This enables precise per-segment latency measurement during inference evaluation.
- `ResNet_model_0` / `ResNet_model_1` — Segmented ResNet variants with intermediate feature extraction.

### Training Entry: KD Pipeline (`main/ResNet/ResNet_KD.py`)

Entry script for training and evaluating the complete EdgeKE pipeline on the ResNet backbone.

**Key components:**
- `Train_ResNet()` — Trains all three branch models independently from scratch (baseline training without KD or early-exit).
- `Train_Teacher(T=1.0)` — Trains the ResNet34 teacher model.
- `KD_ResNet(T, Epoch, Distance_type)` — Performs knowledge distillation on each branch model independently using the pre-trained teacher. Supports KL, JS, and WS distance types.
- `Test_model(type)` — Evaluates a trained BranchyNet with early-exit inference, loading checkpoints and running `Eval_BranchyNet`.

### Training Entry: BranchyNet with KD (`main/ResNet/ResNet_KD_Branch.py`)

Entry script for training BranchyNet models with various strategies, including combined KD + early-exit.

**Key components:**
- `Train_As()` — Asynchronous back-to-front BranchyNet training (shallowest first).
- `Train_Sy()` — Synchronous BranchyNet training with joint optimization.
- `Train_As_main_model()` — Asynchronous training starting from the main model.
- `KD_ResNet(Beta, T, Distance_type, type)` — BranchyNet training with knowledge distillation, combining both EdgeKE design knobs.
- `KD_ResNet_back(...)` — Reverse-order variant of KD-enhanced BranchyNet training.

## Usage

```bash
# Install dependencies
pip install torch torchvision numpy scipy

# Train the teacher model (ResNet34)
cd main/ResNet
python -c "from ResNet_KD import Train_Teacher; Train_Teacher(T=1.0)"

# Train branch models independently (baseline)
python -c "from ResNet_KD import Train_ResNet; Train_ResNet()"

# Knowledge distillation on individual branch models
python -c "from ResNet_KD import KD_ResNet; KD_ResNet(T=1.0, Epoch=400, Distance_type='KL')"

# Synchronous BranchyNet training
python -c "from ResNet_KD_Branch import Train_Sy; Train_Sy()"

# Asynchronous BranchyNet training (back-to-front)
python -c "from ResNet_KD_Branch import Train_As; Train_As()"

# BranchyNet training with Knowledge Distillation (combined approach)
python -c "from ResNet_KD_Branch import KD_ResNet; KD_ResNet(Beta=0.1, T=1.0, Distance_type='KL', type='As')"

# Evaluate a trained BranchyNet with early-exit
python -c "from ResNet_KD import Test_model; Test_model(type='As')"
```

Notes:
- The code automatically detects and uses CUDA-capable GPUs when available.
- Dataset paths are automatically resolved relative to the project root directory via `get_project_dir()`.
- Trained model checkpoints are saved under `model/ResNet/KD/` and `model/ResNet/KD_Branch/`.
- The `Distance_type` parameter in KD functions supports `"KL"` (KL divergence), `"JS"` (JS divergence), and `"WS"` (Wasserstein distance).

## Citation

If you find EdgeKE useful or relevant to your project and research, please kindly cite our paper:

```
@ARTICLE{9294146,
  author={Fang, Weiwei and Xue, Feng and Ding, Yi and Xiong, Naixue and Leung, Victor C. M.},
  journal={IEEE Transactions on Industrial Informatics}, 
  title={EdgeKE: An On-Demand Deep Learning IoT System for Cognitive Big Data on Industrial Edge Devices}, 
  year={2021},
  volume={17},
  number={9},
  pages={6144-6152},
  doi={10.1109/TII.2020.3044930}
}
```

## Contact

Feng Xue (17120431@bjtu.edu.cn)

> Please note that the open source code in this repository was mainly completed by the graduate student author during his master's degree study. Since the author did not continue to engage in scientific research work after graduation, it is difficult to continue to maintain and update these codes. We sincerely apologize that these codes are for reference only.

# Whale Adventure in Fortran: 基于 Fortran 的手写数字识别 CNN

## 项目概述

本项目旨在使用现代 Fortran (f2008 标准) 从零开始构建、训练和评估一个卷积神经网络 (CNN)，并应用于经典的 MNIST 手写数字数据集。项目展示了如何在 Fortran 环境下实现深度学习模型的核心组件，并构建一个完整的训练与验证流程。

## 核心特性

- **模块化设计**: 网络的每一层 (卷积、全连接、激活函数等) 都被封装在独立的 Fortran `module` 中，实现了高内聚和低耦合。
- **动态与可配置**: 模型结构和超参数（如通道数、卷积核大小、学习率等）清晰定义，易于修改和扩展。
- **核心层实现**:
    - `Conv_mod`: 卷积层，支持自定义步长、填充和多通道。
    - `FullConnect_mod`: 全连接层。
    - `PReluFunc_mod`: 参数化 ReLU (PReLU) 激活函数，带有可学习的参数。
    - `Flaten_mod`: 展平层，用于连接卷积层和全连接层。
    - `Dropout_mod`: 在训练期间随机丢弃神经元，防止过拟合。
- **损失函数**:
    - `LossFunc_mod`: 实现了数值稳定的 Softmax 和交叉熵损失函数。
- **数据处理**:
    - `LoadData_mod` & `LoadLabel_mod`: 高效的二进制文件数据加载器，支持按批次 (batch) 读取图像和标签。
    - `DATA_Process/`: 包含用于下载和预处理 MNIST 数据集的 Python 脚本。
- **训练与评估**:
    - `train.f90`: 完整的模型训练脚本，包括前向传播、反向传播、参数更新和周期性的评估。
    - `VisualVal.f90`: 一个可视化工具，用于展示模型在测试集上的预测结果、概率分布以及图像的 ASCII 可视化。

## 网络结构 (`ModelCombine_mod`)

本项目实现了一个经典的 LeNet-style 结构：

1.  **输入**: `28x28x1` 的灰度图像
2.  **Conv1**: 8个 `5x5` 卷积核, 步长 `2`, 填充 `2`
3.  **PReLU1**: 激活函数
4.  **Conv2**: 16个 `5x5` 卷积核, 步长 `2`, 填充 `2`
5.  **PReLU2**: 激活函数
6.  **Flaten**: 将 `7x7x16` 的特征图展平为 `784` 维向量
7.  **FC1**: 全连接层 (`784 -> 128`)
8.  **PReLU3**: 激活函数
9.  **Dropout**: 丢弃率 `p=0.5`
10. **FC2 (输出层)**: 全连接层 (`128 -> 10`)

## 如何使用

### 1. 编译

您可以使用 `gfortran` 编译器来编译项目。请确保所有模块文件 (`.f90`) 都被包含在编译命令中。

**编译训练程序:**
```bash
gfortran -std=f2008 -O3 -o train train.f90 Modules/*.f90
```

**编译可视化验证程序:**
```bash
gfortran -std=f2008 -O3 -o VisualVal VisualVal.f90 Modules/*.f90
```

### 2. 运行

**执行训练:**
```bash
./train
```
训练日志和每个 epoch 的损失、准确率将会被打印到控制台。模型权重会保存在 `RESULTS/Models/` 目录下。

**执行可视化验证:**
```bash
./VisualVal
```
该程序会加载训练好的模型，对一个批次的测试数据进行预测，并以 ASCII 字符画的形式展示输入图像、预测概率和最终结果。

## 项目结构

```
.
├── 1_DATA_Reread/      # 预处理后的二进制 MNIST 数据
├── DATA_Process/       # Python 数据处理脚本
├── Modules/            # 神经网络核心模块
│   ├── Conv_User.f90
│   ├── FullConnect.f90
│   ├── PReluFunc.f90
│   ├── ...
│   └── Test/           # 各模块的单元测试
├── RESULTS/
│   └── Models/         # 保存训练好的模型权重
├── train.f90           # 主训练程序
├── VisualVal.f90       # 可视化验证程序
└── README.md           # 本文档
```

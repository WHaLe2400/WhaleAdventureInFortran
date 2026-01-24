# Whale Adventure in Fortran: 基于 Fortran 的手写数字识别 CNN

## 项目概述

本项目旨在使用现代 Fortran (f2008 标准) 从零开始构建、训练和评估一个卷积神经网络 (CNN)，并应用于经典的 MNIST 手写数字数据集。项目展示了如何在 Fortran 环境下实现深度学习模型的核心组件，并构建一个完整的训练与验证流程。

![模型结构图](https://github.com/WHaLe2400/WhaleAdventureInFortran/blob/main/zzz_IMG/Model_Shape.png)

## 核心特性

- **模块化设计**: 网络的每一层 (卷积、全连接、激活函数等) 都被封装在独立的 Fortran `module` 中，实现了高内聚和低耦合。
- **动态与可配置**: 模型结构和超参数（如通道数、卷积核大小、学习率等）清晰定义，易于修改和扩展。
- **核心层实现**:
    - `Conv_User.f90`: 卷积层，支持自定义步长、填充和多通道。
    - `FullConnect.f90`: 全连接层。
    - `PReluFunc.f90`: 参数化 ReLU (PReLU) 激活函数。
    - `flaten.f90`: 展平层，用于连接卷积层和全连接层。
    - `Dropout.f90`: 在训练期间随机丢弃神经元，防止过拟合。
- **损失函数**:
    - `LossFunc.f90`: 实现了数值稳定的 Softmax 和交叉熵损失函数。
- **数据处理**:
    - `load_data.f90` & `load_label.f90`: 高效的二进制文件数据加载器，支持按批次 (batch) 读取图像和标签。
    - `src/`: 包含用于下载和预处理 MNIST 数据集的 Python 脚本。
- **训练与评估**:
    - `train.f90`: 完整的模型训练脚本，包括前向传播、反向传播、参数更新和周期性的评估。
    - `ValOnly.f90`: 一个纯粹的验证脚本，用于在测试集上计算模型的准确率和损失。
    - `VisualVal.f90`: 一个可视化工具，用于展示模型在测试集上的预测结果、概率分布以及图像的 ASCII 可视化。

## 网络结构 (`ModelCombine.f90`)

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

## 运行效果展示

#### Train_debug:
![训练日志截图](https://github.com/WHaLe2400/WhaleAdventureInFortran/blob/main/zzz_IMG/train_debug.png)

#### ValOnl
![验证结果截图](https://github.com/WHaLe2400/WhaleAdventureInFortran/blob/main/zzz_IMG/ValOnly.png)

#### VisualVal:
![可视化验证截图](https://github.com/WHaLe2400/WhaleAdventureInFortran/blob/main/zzz_IMG/VisualVal.png)

## 如何使用

### 步骤 1: 修改文件路径（重要！）

Fortran 程序中的文件路径是硬编码的。在编译前，您必须根据您当前的项目位置修改这些路径。

需要修改的文件和变量如下：

-   **`train.f90`**:
    -   `train_data_path`: 训练图像数据路径。
    -   `train_label_path`: 训练标签数据路径。
    -   `test_data_path`: 测试图像数据路径。
    -   `test_label_path`: 测试标签数据路径。
    -   `model_save_path`: 模型权重保存的基础路径。

-   **`ValOnly.f90`** 和 **`VisualVal.f90`**:
    -   `test_data_path`: 测试图像数据路径。
    -   `test_label_path`: 测试标签数据路径。
    -   `model_load_path`: 加载已训练模型权重的路径。

**示例**:
假设您的项目完整路径是 `/home/user/WhaleAdventureInFortran/FINAL`，您需要将 `ValOnly.f90` 中的路径修改为：
```fortran
character(len=*), parameter :: file_root = "/home/user/WhaleAdventureInFortran/FINAL/1_DATA_Reread/"
character(len=*), parameter :: model_load_path = "/home/user/WhaleAdventureInFortran/FINAL/ModelWeight"
```

### 步骤 2: 编译源代码

推荐使用 `gfortran` 编译器。

```bash
# 定义编译器和标志
FC=gfortran
FFLAGS="-std=f2008 -O3 -J Modules" # -J 指定 .mod 文件输出目录

# 1. 编译所有模块
${FC} ${FFLAGS} -c Modules/*.f90

# 2. 编译主程序并链接模块
${FC} ${FFLAGS} -o train train.f90 Modules/*.o
${FC} ${FFLAGS} -o ValOnly ValOnly.f90 Modules/*.o
${FC} ${FFLAGS} -o VisualVal VisualVal.f90 Modules/*.o

# 清理中间文件
rm Modules/*.o Modules/*.mod
```

或使用单行命令针对特定文件进行编译：
```bash
# 移动到项目目录
cd <YOUR_PROJECT_PATH>/FINAL

gfortran -std=f2008 -o train train.f90 Modules/*.f90            #针对 train.f90编译指令
gfortran -std=f2008 -o ValOnly ValOnly.f90 Modules/*.f90        #针对 ValOnly.f90编译指令
gfortran -std=f2008 -o VisualVal VisualVal.f90 Modules/*.f90    #针对 VisualVal.f90编译指令
```

### 步骤 3: 运行程序

**执行训练:**
```bash
./train
```
训练日志和每个 epoch 的损失、准确率将会被打印到控制台。模型权重会保存在您在 `train.f90` 中指定的 `model_save_path` 目录下（默认为 `ModelWeight/`）。

**执行验证:**
```bash
./ValOnly
```
该程序会加载训练好的模型，计算并打印在整个测试集上的最终准确率。

**执行可视化验证:**
```bash
./VisualVal
```
该程序会加载模型，对一个批次的测试数据进行预测，并以 ASCII 字符画的形式展示输入图像、预测概率和最终结果。

### 单元测试

项目在 `Modules/Test/` 目录下为每个核心模块提供了单元测试。编译和运行它们是验证各组件是否正常工作的好方法。

**编译并运行单个测试（例如 `ConvTest`）:**
```bash
gfortran -std=f2008 -O3 -o Modules/Test/ConvTest Modules/Test/ConvTest.f90 Modules/Conv_User.f90
./Modules/Test/ConvTest
```

## 项目结构

```
.
├── 1_DATA_Reread/      # 预处理后的二进制 MNIST 数据
├── ModelWeight/        # 保存训练好的模型权重
├── Modules/            # 神经网络核心模块 (.f90 源码)
│   ├── Conv_User.f90
│   ├── FullConnect.f90
│   ├── ...
│   └── Test/           # 各模块的单元测试
├── src/                # Python 数据处理与辅助脚本
├── train.f90           # 主训练程序
├── ValOnly.f90         # 纯验证程序
├── VisualVal.f90       # 可视化验证程序
├── train               # (编译后生成) 训练可执行文件
├── ValOnly             # (编译后生成) 验证可执行文件
├── VisualVal           # (编译后生成) 可视化可执行文件
├── *.mod               # (编译后生成) 模块接口文件
└── README.md           # 本文档
```

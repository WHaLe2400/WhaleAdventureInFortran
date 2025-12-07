import torch
import numpy as np
import os
from Model import CNN  # 从您项目中的 Model.py 导入 CNN 类

def export_conv_layer(layer, filename):
    """
    导出卷积层的权重和偏置。
    PyTorch Conv2d weights: (out_ch, in_ch, kH, kW)
    Fortran ConvLayer weights: (kH, kW, in_ch, out_ch)
    需要进行维度重排。
    """
    print(f"Exporting {filename}...")
    weights = layer.weight.data.numpy()
    biases = layer.bias.data.numpy()

    # 维度重排: (out_ch, in_ch, kH, kW) -> (kH, kW, in_ch, out_ch)
    weights_fortran_order = np.transpose(weights, (2, 3, 1, 0))

    out_ch, in_ch, kH, kW = weights.shape

    with open(filename, 'w') as f:
        # 写入元数据
        f.write(f"{out_ch} {in_ch} {kH} {kW}\n")
        # 写入权重
        # Fortran `read(*, *)` 会按列主序读取，所以我们先 flatten
        f.write(" ".join(map(str, weights_fortran_order.flatten())) + "\n")
        # 写入偏置
        f.write(" ".join(map(str, biases.flatten())) + "\n")
    print("...Done.")

def export_fc_layer(layer, filename):
    """
    导出全连接层的权重和偏置。
    PyTorch Linear weights: (out_features, in_features)
    Fortran FullConnectLayer weights: (output_size, input_size)
    维度匹配，但 Fortran 的加载方式是逐行读取。
    """
    print(f"Exporting {filename}...")
    weights = layer.weight.data.numpy()
    biases = layer.bias.data.numpy()

    with open(filename, 'w') as f:
        # Fortran 代码是逐行读取的，所以我们逐行写入
        for row in weights:
            f.write(" ".join(map(str, row)) + "\n")
        # 写入偏置
        f.write(" ".join(map(str, biases)) + "\n")
    print("...Done.")

def export_prelu_layer(layer, filename):
    """
    导出 PReLU 层的 'a' 参数。
    """
    print(f"Exporting {filename}...")
    weights = layer.weight.data.numpy() # PReLU的权重就是 'a'
    num_channels = len(weights)

    with open(filename, 'w') as f:
        # 写入元数据 (通道数)
        f.write(f"{num_channels}\n")
        # 写入 'a' 参数
        f.write(" ".join(map(str, weights.flatten())) + "\n")
    print("...Done.")

def main():
    # --- 配置 ---
    # 1. 指定 PyTorch 模型权重文件路径
    #    请确保将您训练好的 'mnist_cnn.pt' 文件放置在此脚本所在的 'src' 目录下
    pytorch_model_path = 'mnist_cnn.pt'
    # 2. 指定 Fortran 权重文件的输出目录
    fortran_weights_dir = '../config_fromTroch'

    # --- 加载模型 ---
    print("Loading PyTorch model...")
    model = CNN()
    
    # 如果 PyTorch 权重文件不存在，则创建一个
    if not os.path.exists(pytorch_model_path):
        print(f"'{pytorch_model_path}' not found. Creating a dummy model state.")
        torch.save(model.state_dict(), pytorch_model_path)
        print(f"Dummy '{pytorch_model_path}' created. You should replace it with your trained model.")

    model.load_state_dict(torch.load(pytorch_model_path))
    model.eval() # 设置为评估模式
    print("Model loaded.")

    # --- 创建输出目录 ---
    if not os.path.exists(fortran_weights_dir):
        os.makedirs(fortran_weights_dir)
        print(f"Created directory: {fortran_weights_dir}")

    # --- 导出权重 ---
    print("\nStarting weight export...")
    export_conv_layer(model.conv1, os.path.join(fortran_weights_dir, '_Conv1.dat'))
    export_prelu_layer(model.prelu1, os.path.join(fortran_weights_dir, '_PReLU1.dat'))
    export_conv_layer(model.conv2, os.path.join(fortran_weights_dir, '_Conv2.dat'))
    export_prelu_layer(model.prelu2, os.path.join(fortran_weights_dir, '_PReLU2.dat'))
    export_fc_layer(model.fc1, os.path.join(fortran_weights_dir, '_FC1.dat'))
    export_prelu_layer(model.prelu3, os.path.join(fortran_weights_dir, '_PReLU3.dat'))
    export_fc_layer(model.fc2, os.path.join(fortran_weights_dir, '_FC2.dat'))

    print("\nExport complete!")
    print(f"Fortran weights saved in '{fortran_weights_dir}'")

if __name__ == '__main__':
    main()

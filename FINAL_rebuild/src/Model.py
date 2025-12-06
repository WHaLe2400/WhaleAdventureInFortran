import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 输入: (N, 1, 28, 28)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, stride=2, padding=2)
        # -> (N, 8, 14, 14)
        self.prelu1 = nn.PReLU(num_parameters=8)
        
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=2, padding=2)
        # -> (N, 16, 7, 7)
        self.prelu2 = nn.PReLU(num_parameters=16)
        
        self.flatten = nn.Flatten()
        # -> (N, 16 * 7 * 7) = (N, 784)
        
        self.fc1 = nn.Linear(in_features=16 * 7 * 7, out_features=128)
        # -> (N, 128)
        self.prelu3 = nn.PReLU(num_parameters=128)
        
        self.fc2 = nn.Linear(in_features=128, out_features=10)
        # -> (N, 10)

    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        x = self.prelu2(self.conv2(x))
        x = self.flatten(x)
        x = self.prelu3(self.fc1(x))
        x = self.fc2(x)
        return x

def verify_model_shapes(model, batch_size=64):
    """
    验证模型各层输出的形状是否与图片一致
    """
    print("--- Verifying intermediate shapes: ---")
    # 模拟一个输入张量
    dummy_input = torch.randn(batch_size, 1, 28, 28)
    
    x = model.conv1(dummy_input)
    print(f"- Conv1 output shape: \t{list(x.shape)}")
    x = model.prelu1(x)
    print(f"- PReLU1 output shape: \t{list(x.shape)}")
    
    x = model.conv2(x)
    print(f"- Conv2 output shape: \t{list(x.shape)}")
    x = model.prelu2(x)
    print(f"- PReLU2 output shape: \t{list(x.shape)}")
    
    x = model.flatten(x)
    print(f"- Flatten output shape: \t{list(x.shape)}")
    
    x = model.fc1(x)
    print(f"- FC1 output shape: \t\t{list(x.shape)}")
    x = model.prelu3(x)
    print(f"- PReLU3 output shape: \t{list(x.shape)}")
    
    x = model.fc2(x)
    print(f"- Final output shape: \t{list(x.shape)}")

if __name__ == '__main__':
    # 实例化模型并验证结构
    model = CNN()
    verify_model_shapes(model)
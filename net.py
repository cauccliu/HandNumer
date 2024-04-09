"""
数据集 MNIST N,1*28*28 类别为10,对输出的结果进行激活(0,1)
导入数据集
构建网络
train
test
"""

import torch
from torch import nn

class Net_v1(nn.Module):
    # 初始化模型
    def __init__(self):
        super(Net_v1,self).__init__()
        # 全连接网络
        self.fc_layers=nn.Sequential(
            nn.Linear(1*28*28,100),
            nn.ReLU(),
            nn.Linear(100,100),
            nn.ReLU(),
            nn.Linear(100,50),
            nn.ReLU(),
            nn.Linear(50,32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,10),
            nn.Softmax(dim=1)
        )

    def forward(self,x):
        return self.fc_layers(x)
    
class Net_v2(nn.Module):
    # 初始化模型
    def __init__(self):
        super(Net_v2,self).__init__()
        # 全连接网络
        self.layers=nn.Sequential(
            nn.Conv2d(1,16,(3,3)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16,32,(3,3)),
            nn.ReLU(),
            nn.Conv2d(32,44,(3,3)),
            nn.ReLU(),
            nn.Conv2d(44,64,(3,3)),
            nn.ReLU(),
        )

        self.outlayers = nn.Sequential(
            nn.Linear(64*7*7, 10),
            nn.Softmax(dim=1)
        )

    def forward(self,x):
        out = self.layers(x).reshape(-1,64*7*7)
        out = self.outlayers(out)
        return out
        
    
if __name__ == '__main__':
    """ net = Net_v1()
    x = torch.randn(1,1*28*28) # 批次N=1
    print(net(x).shape) # 输出torch.Size([1, 10]) """

    net = Net_v2()
    x = torch.randn(1,1,28,28) # NCHW
    print(net(x).shape)

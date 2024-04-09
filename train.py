# 导入数据集

from typing import Any
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from net import Net_v1
from net import Net_v2
from torch import nn,optim
import os
from torch.nn.functional import one_hot
import datetime
import torch

train_dataset = datasets.MNIST('./MNIST', train=True, download=True, transform = transforms.ToTensor())
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = datasets.MNIST('./MNIST', train=False, download=True, transform = transforms.ToTensor())
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

DEVICE = 'cuda'

class Train_v1:
    def __init__(self,weight_path):
        self.summaryWriter = SummaryWriter('./logs') # 创建文件夹存放日志

        self.net = Net_v1()  # 实例化Net_v1对象
        self.net = self.net.to(DEVICE)  # 将Net_v1对象移动到设备上
        # self.net = Net_v1.to(DEVICE) # cuda上训练

        if os.path.exists(weight_path):
            self.net.load_state_dict(torch.load(weight_path))
        
        self.opt = optim.Adam(self.net.parameters())
        self.fc_loss = nn.MSELoss()
        self.train = True
        self.test = True

    def __call__(self):
        index1,index2 = 0,0
        for epoch in range(20):
            if self.train:
                for i,(img,label) in enumerate(train_dataloader): # 自动会生成一个索引
                    #print(img.shape) # 64 1 28 28
                    #print(label.shape) # 64 
                    label = one_hot(label,10).float().to(DEVICE) # one-hot 编码，只有一个类别会显示为1，其他的为0 # to(DEVICE)放到cuda上
                    # print(label) # [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]
                    img = img.reshape(-1,1*28*28).to(DEVICE)
                    # print(img.shape) # torch.Size([64, 784])
                    # print(label.shape) # torch.Size([64, 10])
                    train_y = self.net(img) # 将图片输入网络，train_y就是预测值
                    # print(train_y.shape) # torch.Size([64, 10])
                    train_loss = self.fc_loss(train_y,label)
                
                    # 清空梯度
                    self.opt.zero_grad()

                    # 梯度计算
                    train_loss.backward()

                    # 梯度更新
                    self.opt.step()

                    if i%10==0 :
                        print(f"train_loss {i} ===>", train_loss.item())
                        self.summaryWriter.add_scalar('train_loss',train_loss,index1)
                        index1 +=1

                data_time = str(datetime.datetime.now()).replace(' ', '-').replace(':','_').replace('·','_')
                save_dir = 'param'
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                # 保存权重文件
                torch.save(self.net.state_dict(), f'{save_dir}/{data_time}-{epoch}.pt')
                # torch.save(self.net.state_dict(),f'param/{data_time}-{epoch}.pt')# 保存权重

            if self.test:
                # 测试
                for i,(img,label) in enumerate(test_dataloader): # 自动会生成一个索引
                 
                    label = one_hot(label,10).float().to(DEVICE) 
                    img = img.reshape(-1,1*28*28).to(DEVICE)
                
                    test_y = self.net(img) 
                    test_loss = self.fc_loss(test_y,label)
                
                    # 测试不需要梯度计算了

                    # 查看准确率
                    test_y = torch.argmax(test_y,dim=1)
                    label = torch.argmax(label,dim=1)
                    acc = torch.mean(torch.eq(test_y,label).float()) # torch.mean 均值

                    if i%10 ==0:
                        print(f"train_loss {i} ===>", test_loss.item())
                        print(f'acc {i}====>',acc.item())
                        self.summaryWriter.add_scalar('test_loss',test_loss,index2)
                        index2 +=1
                
class Train_v2:
    def __init__(self,weight_path):
        self.summaryWriter = SummaryWriter('./logs') # 创建文件夹存放日志

        self.net = Net_v2()  # 实例化Net_v1对象
        self.net = self.net.to(DEVICE)  # 将Net_v1对象移动到设备上
        # self.net = Net_v1.to(DEVICE) # cuda上训练

        if os.path.exists(weight_path):
            self.net.load_state_dict(torch.load(weight_path))
        
        self.opt = optim.Adam(self.net.parameters())
        self.fc_loss = nn.MSELoss()
        self.train = True
        self.test = True

    def __call__(self):
        index1,index2 = 0,0
        for epoch in range(20):
            if self.train:
                for i,(img,label) in enumerate(train_dataloader): # 自动会生成一个索引
                    label = one_hot(label,10).float().to(DEVICE) # one-hot 编码，只有一个类别会显示为1，其他的为0 # to(DEVICE)放到cuda上
                    # img = img.reshape(-1,1*28*28).to(DEVICE) # 图像本身就是3维的，不需要再reshape
                    # print(img.shape) # torch.Size([64, 1, 28, 28])
                    img = img.to(DEVICE)
                    train_y = self.net(img) 
                    train_loss = self.fc_loss(train_y,label)
                
                    # 清空梯度
                    self.opt.zero_grad()
                    # 梯度计算
                    train_loss.backward()
                    # 梯度更新
                    self.opt.step()

                    if i%10==0 :
                        print(f"train_loss {i} ===>", train_loss.item())
                        self.summaryWriter.add_scalar('train_loss',train_loss,index1)
                        index1 +=1

                data_time = str(datetime.datetime.now()).replace(' ', '-').replace(':','_').replace('·','_')
                save_dir = 'param'
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                # 保存权重文件
                torch.save(self.net.state_dict(), f'{save_dir}/{data_time}-{epoch}.pt')
                # torch.save(self.net.state_dict(),f'param/{data_time}-{epoch}.pt')# 保存权重

            if self.test:
                # 测试
                for i,(img,label) in enumerate(test_dataloader): # 自动会生成一个索引
                 
                    label = one_hot(label,10).float().to(DEVICE) 
                    # img = img.reshape(-1,1*28*28).to(DEVICE)
                    img = img.to(DEVICE)
                    test_y = self.net(img) 
                    test_loss = self.fc_loss(test_y,label)
                
                    # 测试不需要梯度计算了

                    # 查看准确率
                    test_y = torch.argmax(test_y,dim=1)
                    label = torch.argmax(label,dim=1)
                    acc = torch.mean(torch.eq(test_y,label).float()) # torch.mean 均值

                    if i%10 ==0:
                        print(f"train_loss {i} ===>", test_loss.item())
                        print(f'acc {i}====>',acc.item())
                        self.summaryWriter.add_scalar('test_loss',test_loss,index2)
                        index2 +=1

if __name__ == '__main__':
    """ train=Train_v1('param/1.pt')
    train.__call__() # 调用函数 """

    train2=Train_v2('param/1.pt')
    train2.__call__() # 调用函数
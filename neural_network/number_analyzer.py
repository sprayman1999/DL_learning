#!/bin/python3
import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn             # neural network神经网络包（核心）
import torch.nn.functional as F   # F中包含激活函数
import torch.utils.data as Data   # Data是批训练的模块
import torchvision                # 包含计算机视觉的相关图片库
EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = True
CESHI = 2000
train_data = torchvision.datasets.MNIST(
        root='./mnist',
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=DOWNLOAD_MNIST)

test_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=False
)

# 插入新的维度
test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:CESHI] / 255
test_y = test_data.targets[:CESHI]

# 查看数据集
print(train_data.data.size())
print(train_data.targets.size())
print(test_data.data.size())
print(test_data.targets.size())

# 3.3设置批训练数据
train_loader = Data.DataLoader(
    dataset=train_data,           # 数据集
    batch_size=BATCH_SIZE,        # 一批训练数据的个数
    shuffle=True,                 # 下次训练是否打乱数据顺序
    num_workers=2                 # 多线程，使用双进程提取数据
)

# 4.建立CNN网络
class CNN(nn.Module):
    # 4.1 设置神经网络属性，定义各层信息
    def __init__(self):
        super(CNN, self).__init__()         #继承Module的固定格式
        
        '''
        torch.nn.Conv2d(
            in_channels,#输入的通道数 输入（bachsize，c，w，h）中的c
            out_channels,               #输出通道数  输出（bachsize，c，w，h）中的c
            kernel_size,                #卷积核的大小
            stride=1,                   #步长
            padding=0,                  #补偿在边上补白，填充
            dilation=1,                 #用于空洞卷积？后续补充
            groups=1,                   #分group进行卷积，每个卷积看到的东西不太一样
            bias=True,                  #是否有偏置，也就是wx+b的b那种常熟偏置
            padding_mode='zeros',       #pading的时候补充的值是否为0
            device=None,                #卷积核在哪个设备上"cuda:0" or "cpu"
            dtype=None
        )                 #类型？后续补充
        '''
        # 建立卷积层1
        self.conv1 = nn.Sequential(
            nn.Conv2d(                      # 卷积层
                in_channels=1,              # 输入图片的高度（灰度图1,彩色图3）
                out_channels=16,            # 输出图片的高度（提取feature的个数）
                kernel_size=5,              # 过滤器的单次扫描面积5×5
                stride=1,                   # 步长
                padding=2                   # 在输入图片周围补0,补全图片的边缘像素点
            ),                              # 图片大小(1,28,28)->(16,28,28)
            nn.ReLU(),                      # 激活函数
            nn.MaxPool2d(                   # 池化层
                kernel_size=2               # 压缩长和宽，取2×2像素区域的最大值作为该区域值
            )                               # 图片大小(16,28,28)->(16,14,14)
        )
        # 建立卷积层2
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),     # 图片大小(16,14,14)->(32,14,14)
            nn.ReLU(),
            nn.MaxPool2d(2)                 # 图片大小(32,14,14)->(32,7,7)
        )
        # 建立输出层
        self.out = nn.Linear(32*7*7, 10)
    # 4.2 前向传播过程
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)             # (batch,32,7,7)
        x = x.view(x.size(0), -1)     # (batch,32*7*7) 将三维数据展平成二维数据
        output = self.out(x)
        return output


def test():
    cnn = CNN()
    # 5.优化器
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # 优化网络的所有参数
    loss_func = nn.CrossEntropyLoss()                      # 误差计算函数
    # 6.训练
    for epoch in range(EPOCH):
        for step, (x, y) in enumerate(train_loader):
            output = cnn(x)
            loss = loss_func(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 可视化训练过程
            if step % 50 == 0:
                test_output = cnn(test_x)
                pred_y = torch.max(test_output, 1)[1].data.squeeze()
                accuracy = (pred_y == test_y).sum().item() / CESHI
                print('Epoch:', epoch,'|loss:', loss.data.numpy(),'|accuracy:', accuracy)
    # 7.测试
    test_output = cnn(test_x[:10])                        # 取前测试集的10图片测试
    pred_y = torch.max(test_output, 1)[1].data.squeeze()  
    print('真实值：', test_y[:10].numpy())                 # 真实值
    print('预测值：', pred_y.numpy())                      # 预测值

    # 把前10张图片打印出来
    plt.figure(1, figsize=(10, 3))
    for i in range(1, 11):
        plt.subplot(1, 10, i)
        plt.imshow(test_data.data[i-1].numpy(), cmap='gray')
        plt.axis('off')                   # 去除图片的坐标轴
        plt.xticks([])                    # 去除x轴刻度
        plt.yticks([])                    # 去除y轴刻度
    plt.show()

    # 保存训练好的网络
    torch.save(cnn.state_dict(), 'cnn_params.pkl')


def main():
    test()
if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# -------------------------- 1. 数据预处理与加载 --------------------------
# 数据预处理：归一化到 [0,1]，转为张量
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 均值和标准差
])

# 加载 MNIST 数据集
train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

# 数据加载器
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# -------------------------- 2. GoogLeNet 实现（含 InceptionA 模块） --------------------------
class InceptionA(nn.Module):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        # 分支1：1x1 卷积（直接降维）
        self.branch1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)

        # 分支2：1x1 卷积（降维）→ 5x5 卷积（特征提取）
        self.branch5x5_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2)  # padding=2 保持尺寸

        # 分支3：1x1 卷积（降维）→ 3x3 卷积 → 3x3 卷积（多尺度特征）
        self.branch3x3_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)

        # 分支4：平均池化 → 1x1 卷积（降维）
        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)  # 保持尺寸
        branch_pool = self.branch_pool(branch_pool)

        # 拼接所有分支（维度1：通道数）
        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        return torch.cat(outputs, dim=1)


class GoogLeNetMNIST(nn.Module):
    def __init__(self):
        super(GoogLeNetMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)  # 输入1通道（灰度图），输出10通道
        self.conv2 = nn.Conv2d(88, 20, kernel_size=5)  # InceptionA输出通道数：16+24+24+24=88
        self.incep1 = InceptionA(in_channels=10)  # conv1输出10通道，作为InceptionA输入
        self.incep2 = InceptionA(in_channels=20)  # conv2输出20通道，作为InceptionA输入
        self.mp = nn.MaxPool2d(2)  # 最大池化（步长2，尺寸减半）
        self.fc = nn.Linear(1408, 10)  # 全连接层：输入特征数1408，输出10类（0-9）

    def forward(self, x):
        in_size = x.size(0)  # batch_size
        # 第一层：卷积 → 池化 → ReLU
        x = F.relu(self.mp(self.conv1(x)))
        # 第一个Inception模块
        x = self.incep1(x)
        # 第二层：卷积 → 池化 → ReLU
        x = F.relu(self.mp(self.conv2(x)))
        # 第二个Inception模块
        x = self.incep2(x)
        # 展平特征图（batch_size, 特征数）
        x = x.view(in_size, -1)
        # 全连接层输出
        x = self.fc(x)
        return x


# -------------------------- 3. ResNet 实现（含残差块） --------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        # 残差块：2个3x3卷积（保持通道数和尺寸不变）
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        # 残差路径：conv1 → ReLU → conv2
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        #  shortcut 连接：原始输入 + 残差输出 → ReLU
        return F.relu(x + y)


class ResNetMNIST(nn.Module):
    def __init__(self):
        super(ResNetMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)  # 输入1通道，输出16通道
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)  # 输入16通道，输出32通道
        self.mp = nn.MaxPool2d(2)  # 最大池化
        self.rblock1 = ResidualBlock(16)  # 输入16通道的残差块
        self.rblock2 = ResidualBlock(32)  # 输入32通道的残差块
        self.fc = nn.Linear(512, 10)  # 全连接层：输入512，输出10类

    def forward(self, x):
        in_size = x.size(0)
        # 第一层：卷积 → ReLU → 池化
        x = self.mp(F.relu(self.conv1(x)))
        # 第一个残差块
        x = self.rblock1(x)
        # 第二层：卷积 → ReLU → 池化
        x = self.mp(F.relu(self.conv2(x)))
        # 第二个残差块
        x = self.rblock2(x)
        # 展平特征图：32通道 × 4×4尺寸 = 512
        x = x.view(in_size, -1)
        # 全连接层输出
        x = self.fc(x)
        return x


# -------------------------- 4. 训练与测试函数 --------------------------
def train(model, device, train_loader, optimizer, epoch, log_interval=300):
    model.train()  # 训练模式
    train_losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  # 梯度清零
        output = model(data)  # 前向传播
        loss = F.cross_entropy(output, target)  # 交叉熵损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        # 记录损失
        if batch_idx % log_interval == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
            train_losses.append(loss.item())
    return train_losses


def test(model, device, test_loader):
    model.eval()  # 测试模式
    test_loss = 0
    correct = 0
    with torch.no_grad():  # 禁用梯度计算
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # 累计损失
            pred = output.argmax(dim=1, keepdim=True)  # 预测类别
            correct += pred.eq(target.view_as(pred)).sum().item()  # 累计正确数

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(
        f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return test_loss, accuracy


# -------------------------- 5. 主运行流程 --------------------------
if __name__ == '__main__':
    # 设置设备（GPU优先，备用CPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 超参数设置
    epochs = 10  # 训练轮数
    lr = 0.01  # 学习率
    momentum = 0.9  # 动量（SGD优化器）

    # -------------------------- 训练 GoogLeNet --------------------------
    print("=" * 50)
    print("Training GoogLeNet...")
    print("=" * 50)
    googlenet = GoogLeNetMNIST().to(device)
    optimizer_googlenet = torch.optim.SGD(googlenet.parameters(), lr=lr, momentum=momentum)

    # 记录训练过程
    googlenet_train_losses = []
    googlenet_test_losses = []
    googlenet_accuracies = []

    for epoch in range(1, epochs + 1):
        train_loss = train(googlenet, device, train_loader, optimizer_googlenet, epoch)
        test_loss, accuracy = test(googlenet, device, test_loader)

        googlenet_train_losses.extend(train_loss)
        googlenet_test_losses.append(test_loss)
        googlenet_accuracies.append(accuracy)

    # -------------------------- 训练 ResNet --------------------------
    print("=" * 50)
    print("Training ResNet...")
    print("=" * 50)
    resnet = ResNetMNIST().to(device)
    optimizer_resnet = torch.optim.SGD(resnet.parameters(), lr=lr, momentum=momentum)

    # 记录训练过程
    resnet_train_losses = []
    resnet_test_losses = []
    resnet_accuracies = []

    for epoch in range(1, epochs + 1):
        train_loss = train(resnet, device, train_loader, optimizer_resnet, epoch)
        test_loss, accuracy = test(resnet, device, test_loader)

        resnet_train_losses.extend(train_loss)
        resnet_test_losses.append(test_loss)
        resnet_accuracies.append(accuracy)

    # -------------------------- 可视化结果 --------------------------
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
    plt.figure(figsize=(12, 4))

    # 1. 测试准确率对比
    plt.subplot(1, 2, 1)
    epochs_range = range(1, epochs + 1)
    plt.plot(epochs_range, googlenet_accuracies, label='GoogLeNet', marker='o')
    plt.plot(epochs_range, resnet_accuracies, label='ResNet', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title('准确率对比')
    plt.legend()
    plt.grid(True)

    # 2. 测试损失对比
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, googlenet_test_losses, label='GoogLeNet', marker='o')
    plt.plot(epochs_range, resnet_test_losses, label='ResNet', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Test Loss')
    plt.title('损失对比')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300)
    plt.show()

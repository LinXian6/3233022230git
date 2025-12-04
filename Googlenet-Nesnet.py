import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt  # 导入可视化库

# -------------------------- 1. 数据加载（MNIST）--------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST均值和标准差
])

train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=2)


# -------------------------- 2. ResNet实现（基于你的代码完善）--------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        # 主路径
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # shortcut路径（处理通道数/尺寸不匹配）
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # 残差连接
        out = F.relu(out)
        return out


class ResNet_MNIST(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet_MNIST, self).__init__()
        self.in_channels = 16
        # 初始卷积层（MNIST输入：1x28x28）
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.mp = nn.MaxPool2d(2)

        # 残差块（2个残差块组）
        self.layer1 = self._make_layer(16, 16, stride=1)  # 输出：16x14x14（经过maxpool后）
        self.layer2 = self._make_layer(16, 32, stride=1)  # 输出：32x7x7（经过第二次maxpool后）

        # 全连接层（32通道×7×7=1568，修正原代码的512错误）
        self.fc = nn.Linear(32 * 7 * 7, num_classes)

    def _make_layer(self, in_channels, out_channels, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        # 输入：x = [64, 1, 28, 28]（batch_size=64）
        out = F.relu(self.bn1(self.conv1(x)))  # [64, 16, 28, 28]
        out = self.mp(out)  # [64, 16, 14, 14]
        out = self.layer1(out)  # [64, 16, 14, 14]
        out = F.relu(self.conv2(out)) if hasattr(self, 'conv2') else out  # 兼容原结构
        out = self.mp(out)  # [64, 32, 7, 7]（修正通道数）
        out = self.layer2(out)  # [64, 32, 7, 7]
        out = out.view(out.size(0), -1)  # [64, 32*7*7=1568]
        out = self.fc(out)  # [64, 10]
        return out


# -------------------------- 3. GoogleNet（Inception v1）实现（适配MNIST）--------------------------
class InceptionBlock(nn.Module):
    def __init__(self, in_channels, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, pool_proj):
        super(InceptionBlock, self).__init__()
        # 1x1卷积分支
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, n1x1, kernel_size=1, bias=False),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(inplace=True)
        )

        # 3x3卷积分支（先1x1降维）
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, n3x3_reduce, kernel_size=1, bias=False),
            nn.BatchNorm2d(n3x3_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(n3x3_reduce, n3x3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(inplace=True)
        )

        # 5x5卷积分支（先1x1降维）
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, n5x5_reduce, kernel_size=1, bias=False),
            nn.BatchNorm2d(n5x5_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(n5x5_reduce, n5x5, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(inplace=True)
        )

        # 池化分支（maxpool + 1x1卷积）
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1, bias=False),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 拼接4个分支的输出（通道维度）
        out = torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)
        return out


class GoogleNet_MNIST(nn.Module):
    def __init__(self, num_classes=10):
        super(GoogleNet_MNIST, self).__init__()
        # 初始卷积层（适配MNIST 1x28x28）
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 输出：16x14x14
        )

        # Inception模块组（简化版，适配小尺寸特征图）
        self.inception1 = InceptionBlock(16, 6, 6, 12, 2, 4, 4)  # 输出通道：6+12+4+4=26
        self.inception2 = InceptionBlock(26, 10, 8, 16, 3, 6, 6)  # 输出通道：10+16+6+6=38
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)  # 输出：38x7x7

        # 全局平均池化 + 全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(38, num_classes)

    def forward(self, x):
        # 输入：[64, 1, 28, 28]
        x = self.stem(x)  # [64, 16, 14, 14]
        x = self.inception1(x)  # [64, 26, 14, 14]
        x = self.inception2(x)  # [64, 38, 14, 14]
        x = self.maxpool(x)  # [64, 38, 7, 7]
        x = self.avgpool(x)  # [64, 38, 1, 1]
        x = x.view(x.size(0), -1)  # [64, 38]
        x = self.dropout(x)
        x = self.fc(x)  # [64, 10]
        return x


# -------------------------- 4. 训练和评估函数 --------------------------
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    train_loss /= len(train_loader.dataset)
    train_acc = 100. * correct / len(train_loader.dataset)
    print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    return train_loss, train_acc


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
    return test_loss, test_acc


# -------------------------- 5. 可视化函数 --------------------------
def plot_results(results, epochs):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
    plt.rcParams['axes.unicode_minus'] = False  # 支持负号

    # 创建2x2子图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('ResNet vs GoogleNet 性能对比（MNIST）', fontsize=16)

    # 定义颜色和标签
    colors = {'ResNet': '#1f77b4', 'GoogleNet': '#ff7f0e'}
    epochs_range = np.arange(1, epochs + 1)

    # 1. 训练损失对比
    ax1 = axes[0, 0]
    for name, res in results.items():
        ax1.plot(epochs_range, res['train_loss'], color=colors[name], label=name, linewidth=2)
    ax1.set_title('训练损失曲线', fontsize=14)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('损失值', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 测试损失对比
    ax2 = axes[0, 1]
    for name, res in results.items():
        ax2.plot(epochs_range, res['test_loss'], color=colors[name], label=name, linewidth=2)
    ax2.set_title('测试损失曲线', fontsize=14)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('损失值', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 训练准确率对比
    ax3 = axes[1, 0]
    for name, res in results.items():
        ax3.plot(epochs_range, res['train_acc'], color=colors[name], label=name, linewidth=2)
    ax3.set_title('训练准确率曲线', fontsize=14)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('准确率（%）', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(90, 100)  # 缩小y轴范围，更清晰展示差异

    # 4. 测试准确率对比
    ax4 = axes[1, 1]
    for name, res in results.items():
        ax4.plot(epochs_range, res['test_acc'], color=colors[name], label=name, linewidth=2)
    ax4.set_title('测试准确率曲线', fontsize=14)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('准确率（%）', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(90, 100)  # 缩小y轴范围，更清晰展示差异

    # 调整布局
    plt.tight_layout()
    # 保存图片（可选）
    plt.savefig('resnet_googlenet_comparison.png', dpi=300, bbox_inches='tight')
    # 显示图片
    plt.show()


# -------------------------- 6. 主实验流程 --------------------------
def main():
    # 设备配置（GPU优先）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # 超参数（两个模型保持一致）
    epochs = 15
    lr = 0.01
    momentum = 0.9
    weight_decay = 5e-4

    # 初始化模型、优化器
    models = {
        'ResNet': ResNet_MNIST().to(device),
        'GoogleNet': GoogleNet_MNIST().to(device)
    }

    optimizers = {
        name: optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        for name, model in models.items()
    }

    # 记录性能（移除time字段）
    results = {name: {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []} for name in models}

    # 训练每个模型
    for name, model in models.items():
        print(f'\n{"=" * 50} Training {name} {"=" * 50}')
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train(model, device, train_loader, optimizers[name], epoch)
            test_loss, test_acc = test(model, device, test_loader)
            results[name]['train_loss'].append(train_loss)
            results[name]['train_acc'].append(train_acc)
            results[name]['test_loss'].append(test_loss)
            results[name]['test_acc'].append(test_acc)

    # -------------------------- 性能对比总结 --------------------------
    print(f'\n{"=" * 60} 性能对比总结 {"=" * 60}')
    print(f'{"模型":<12} {"最终测试准确率":<15} {"参数量(万)":<12}')
    print('-' * 80)

    for name, model in models.items():
        # 计算参数量（万）
        params = sum(p.numel() for p in model.parameters()) / 1e4
        # 最终测试准确率
        final_test_acc = results[name]['test_acc'][-1]
        print(f'{name:<12} {final_test_acc:<15.2f}% {params:<12.2f}')

    # -------------------------- 可视化结果 --------------------------
    plot_results(results, epochs)


if __name__ == '__main__':
    main()
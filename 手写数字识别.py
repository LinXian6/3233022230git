import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# 设置随机种子以保证结果可重复
torch.manual_seed(42)
np.random.seed(42)


# ---------------------- 定义两个模型 ----------------------
# 全连接神经网络（FC）
class FCNet(nn.Module):
    def __init__(self):
        super(FCNet, self).__init__()
        self.l1 = nn.Linear(784, 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, 128)
        self.l4 = nn.Linear(128, 64)
        self.l5 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)  # 展平为784维向量
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)


# 卷积神经网络（CNN）
class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.25)  # 添加dropout防止过拟合
        self.fc = nn.Linear(320, 10)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.pool(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.pool(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(batch_size, -1)  # 展平
        x = self.fc(x)
        return x


# ---------------------- 数据加载 ----------------------
def load_data(batch_size=64):
    # 数据预处理：归一化到[0,1]，转换为Tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的均值和标准差
    ])

    # 加载MNIST数据集
    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return train_loader, test_loader


# ---------------------- 训练函数 ----------------------
def train_model(model, train_loader, test_loader, device, epochs=10, lr=0.001):
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 记录训练过程中的准确率
    train_acc_history = []
    test_acc_history = []

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 计算训练准确率
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_acc = 100 * train_correct / train_total
        train_acc_history.append(train_acc)

        # 测试阶段
        model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():  # 禁用梯度计算
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        test_acc = 100 * test_correct / test_total
        test_acc_history.append(test_acc)

        # 打印每个epoch的结果
        print(f'Epoch [{epoch + 1}/{epochs}], Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')

    return train_acc_history, test_acc_history


# ---------------------- 主函数 ----------------------
if __name__ == "__main__":
    # 设置设备（GPU优先）
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # 加载数据
    batch_size = 64
    train_loader, test_loader = load_data(batch_size)

    # 初始化两个模型并移到设备上
    fc_model = FCNet().to(device)
    cnn_model = CNNNet().to(device)

    # 训练参数
    epochs = 15  # 训练15个epoch，足够看出差异
    lr = 0.001

    # 训练FC模型
    print("\n=== Training Fully Connected Network ===")
    fc_train_acc, fc_test_acc = train_model(fc_model, train_loader, test_loader, device, epochs, lr)

    # 训练CNN模型
    print("\n=== Training Convolutional Neural Network ===")
    cnn_train_acc, cnn_test_acc = train_model(cnn_model, train_loader, test_loader, device, epochs, lr)

    # ---------------------- 绘制准确率对比图 ----------------------
    plt.figure(figsize=(12, 5))

    # 子图1：训练准确率对比
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), fc_train_acc, label='FC Network', marker='o', linewidth=2)
    plt.plot(range(1, epochs + 1), cnn_train_acc, label='CNN', marker='s', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Training Accuracy (%)')
    plt.title('Training Accuracy Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 子图2：测试准确率对比
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), fc_test_acc, label='FC Network', marker='o', linewidth=2)
    plt.plot(range(1, epochs + 1), cnn_test_acc, label='CNN', marker='s', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test Accuracy Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 调整布局并保存/显示图片
    plt.tight_layout()
    plt.savefig('mnist_acc_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 输出最终准确率对比
    print("\n=== Final Accuracy Comparison ===")
    print(f'FC Network Test Accuracy: {fc_test_acc[-1]:.2f}%')
    print(f'CNN Test Accuracy: {cnn_test_acc[-1]:.2f}%')
    print(f'Performance Gap: {cnn_test_acc[-1] - fc_test_acc[-1]:.2f} percentage points')
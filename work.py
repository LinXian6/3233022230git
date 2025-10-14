import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# 读取数据集
try:
    data = pd.read_csv('countries.csv')
    print(f"✅ 成功读取数据集，共 {len(data)} 行原始数据")
except FileNotFoundError:
    raise FileNotFoundError("❌ 未找到 countries.csv 文件，请确认文件路径正确！")


# 定义数据集类（含缺失值删除+数据类型转换）
class MyDataset(Dataset):
    def __init__(self, data):
        # 删除包含 nan 值的行
        self.data = data.dropna()
        print(f"✅ 删除缺失值后，剩余有效数据 {len(self.data)} 行")

        # 特征列（7个输入特征）和目标列（总生态足迹）
        self.feat_cols = ['Population (millions)', 'HDI', 'Cropland Footprint',
                          'Grazing Footprint', 'Forest Footprint', 'Carbon Footprint',
                          'Fish Footprint']
        self.x = self.data[self.feat_cols].values
        self.y = self.data['Total Ecological Footprint'].values

        # 转换为PyTorch张量（float32类型，避免精度问题）
        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32).unsqueeze(1)  # 扩展为列向量

        # 输出数据基本信息，方便排查异常
        print(f"📊 特征数据形状: {self.x.shape}, 目标数据形状: {self.y.shape}")
        print(f"📈 目标值范围: {self.y.min().item():.2f} ~ {self.y.max().item():.2f}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# 实例化数据集并分割训练/测试集（8:2分割）
dataset = MyDataset(data)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# 数据加载器（批量处理+打乱）
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
print(f"🔧 训练集批次数量: {len(train_loader)} (每批16个样本)")
print(f"🔧 测试集批次数量: {len(test_loader)} (每批16个样本)")



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 输入层(7)→隐藏层1(7)→隐藏层2(6)→隐藏层3(5)→输出层(1)
        self.fc1 = nn.Linear(7, 7)  # 第1层：7输入→7输出
        self.fc2 = nn.Linear(7, 6)  # 第2层：7输入→6输出
        self.fc3 = nn.Linear(6, 5)  # 第3层：6输入→5输出
        self.fc4 = nn.Linear(5, 1)  # 输出层：5输入→1输出（回归任务无激活）
        self.relu = nn.ReLU()  # ReLU激活函数（统一定义，避免重复）

    def forward(self, x):
        # 前向传播：输入→ReLU→输入→ReLU→输入→ReLU→输出
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x



# 实例化模型、损失函数（MSE）、优化器（Adam）
model = Net()
criterion = nn.MSELoss()  # 回归任务常用MSE损失
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器（比SGD更稳定）

# 训练参数
num_epochs = 100
train_losses = []  # 存储训练损失
test_losses = []  # 存储测试损失
best_loss = float('inf')  # 记录最佳测试损失（用于保存最优模型）

# 初始化Matplotlib图像（设置样式，支持实时更新）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 支持英文标签
plt.rcParams['axes.unicode_minus'] = False  # 支持负号显示
fig, ax = plt.subplots(figsize=(10, 6))  # 设置图像大小
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('MSE Loss', fontsize=12)
ax.set_title('Training vs Test Loss (5-Layer Neural Network)', fontsize=14, pad=20)
ax.grid(True, alpha=0.3)  # 添加网格线，方便读数

# 训练循环
for epoch in range(num_epochs):
    model.train()  # 开启训练模式（启用Dropout等，此处无但规范保留）
    running_train_loss = 0.0
    for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
        # 梯度清零→前向传播→计算损失→反向传播→参数更新
        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()  # 累加批次损失

    # 计算本轮训练平均损失
    avg_train_loss = running_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # -------------------------- 测试阶段 --------------------------
    model.eval()  # 开启评估模式（禁用Dropout等）
    running_test_loss = 0.0
    with torch.no_grad():  # 禁用梯度计算，加速并避免内存占用
        for x_batch, y_batch in test_loader:
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            running_test_loss += loss.item()

    # 计算本轮测试平均损失
    avg_test_loss = running_test_loss / len(test_loader)
    test_losses.append(avg_test_loss)

    # -------------------------- 保存最优模型 --------------------------
    if avg_test_loss < best_loss:
        best_loss = avg_test_loss
        torch.save(model.state_dict(), 'best_model.pt')
        print(f"📌 Epoch {epoch}: 测试损失下降至 {best_loss:.4f}，保存最优模型")


    if epoch % 10 == 0:
        print(f"Epoch [{epoch:3d}/{num_epochs}]: "
              f"Train Loss = {avg_train_loss:.4f}, "
              f"Test Loss = {avg_test_loss:.4f}")

    if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
        ax.clear()  # 清空上一轮图像
        # 绘制训练/测试损失曲线（添加曲线标签）
        ax.plot(range(1, epoch + 2), train_losses,
                label='Train Loss', color='#2E86AB', linewidth=2.5, marker='o', markersize=3)
        ax.plot(range(1, epoch + 2), test_losses,
                label='Test Loss', color='#A23B72', linewidth=2.5, marker='s', markersize=3)
        # 标注最佳测试损失点
        best_epoch = test_losses.index(best_loss) + 1
        ax.scatter(best_epoch, best_loss, color='red', s=80, zorder=5,
                   label=f'Best Test Loss: {best_loss:.4f} (Epoch {best_epoch})')
        # 重新设置标签和网格
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('MSE Loss', fontsize=12)
        ax.set_title('Training vs Test Loss (5-Layer Neural Network)', fontsize=14, pad=20)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        plt.pause(0.1)  # 暂停0.1秒，让图像更新


plt.tight_layout()  # 自动调整布局，避免标签被截断
plt.savefig('training_test_loss.png', dpi=300, bbox_inches='tight')  # 高分辨率保存
plt.show()


print("\n" + "=" * 50)
print("训练完成！")
print(f"📊 最终训练损失: {train_losses[-1]:.4f}")
print(f"📊 最终测试损失: {test_losses[-1]:.4f}")
print(f"🏆 最佳测试损失: {best_loss:.4f} (对应Epoch {best_epoch})")
print(f"💾 最优模型已保存至: best_model.pt")
print(f"💾 损失可视化图已保存至: training_test_loss.png")
print("=" * 50)
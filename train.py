import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns  # 用于绘制混淆矩阵（需提前安装：pip install seaborn）

# ---------------------- 1. 加载与预处理数据 ----------------------
# 加载数据集
data = pd.read_csv('health_lifestyle_dataset.csv')

# 对 gender 进行编码
label_encoder = LabelEncoder()
data['gender'] = label_encoder.fit_transform(data['gender'])  # 0=Female, 1=Male

# 划分特征和目标变量
X = data.drop(['id', 'disease_risk'], axis=1).values
y = data['disease_risk'].values
feature_names = data.drop(['id', 'disease_risk'], axis=1).columns  # 保存特征名用于可视化

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为张量
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)


# ---------------------- 2. 自定义数据集与加载器 ----------------------
class HealthDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = HealthDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = HealthDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# ---------------------- 3. 定义全连接神经网络 ----------------------
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, len(np.unique(y)))

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out


# ---------------------- 4. 初始化模型与训练（记录训练过程数据） ----------------------
input_size = X_train.shape[1]
model = NeuralNetwork(input_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 记录训练过程的关键指标（用于可视化）
train_loss_history = []  # 每轮训练损失
test_loss_history = []  # 每轮测试损失
test_acc_history = []  # 每轮测试准确率

num_epochs = 10
for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    running_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_train_loss = running_loss / len(train_loader)
    train_loss_history.append(avg_train_loss)

    # 测试阶段（每轮训练后评估测试集）
    model.eval()
    test_running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []  # 记录所有测试集预测结果（用于后续混淆矩阵）
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            # 计算测试损失
            loss = criterion(outputs, batch_y)
            test_running_loss += loss.item()
            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            # 收集预测结果
            all_preds.extend(predicted.numpy())

    avg_test_loss = test_running_loss / len(test_loader)
    test_acc = correct / total
    test_loss_history.append(avg_test_loss)
    test_acc_history.append(test_acc)

    # 打印每轮结果
    print(f'Epoch {epoch + 1}/{num_epochs}')
    print(f'Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | Test Acc: {test_acc:.4f}\n')

# ---------------------- 5. 可视化模块（4类关键图表） ----------------------
# 设置中文字体（避免中文乱码）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
fig, axes = plt.subplots(2, 2, figsize=(16, 12))  # 2行2列子图，总大小16x12
fig.suptitle('健康生活方式数据集 - 神经网络模型分析', fontsize=16, fontweight='bold')

# 子图1：训练/测试损失曲线（查看训练收敛情况）
axes[0, 0].plot(range(1, num_epochs + 1), train_loss_history, label='训练损失', color='#1f77b4', linewidth=2.5,
                marker='o')
axes[0, 0].plot(range(1, num_epochs + 1), test_loss_history, label='测试损失', color='#ff7f0e', linewidth=2.5,
                marker='s')
axes[0, 0].set_xlabel('训练轮次（Epoch）', fontsize=12)
axes[0, 0].set_ylabel('损失值（Loss）', fontsize=12)
axes[0, 0].set_title('训练与测试损失变化曲线', fontsize=14, fontweight='bold')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3)  # 添加网格线，增强可读性

# 子图2：测试集准确率曲线（查看模型性能变化）
axes[0, 1].plot(range(1, num_epochs + 1), test_acc_history, color='#2ca02c', linewidth=3, marker='D', markersize=6)
axes[0, 1].set_xlabel('训练轮次（Epoch）', fontsize=12)
axes[0, 1].set_ylabel('测试集准确率', fontsize=12)
axes[0, 1].set_title('测试集准确率变化曲线', fontsize=14, fontweight='bold')
axes[0, 1].set_ylim(0.5, 1.0)  # 限定y轴范围（0.5-1.0），突出变化
axes[0, 1].grid(True, alpha=0.3)
# 在每个点标注准确率数值
for i, acc in enumerate(test_acc_history):
    axes[0, 1].annotate(f'{acc:.3f}', (i + 1, acc), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)

# 子图3：目标变量（disease_risk）分布（查看数据平衡性）
# 统计训练集和测试集的类别分布
train_class_count = np.bincount(y_train)
test_class_count = np.bincount(y_test)
classes = np.unique(y)  # 所有类别
x_pos = np.arange(len(classes))

# 绘制双柱状图
width = 0.35
axes[1, 0].bar(x_pos - width / 2, train_class_count, width, label='训练集', color='#d62728', alpha=0.8)
axes[1, 0].bar(x_pos + width / 2, test_class_count, width, label='测试集', color='#9467bd', alpha=0.8)
axes[1, 0].set_xlabel('疾病风险类别（disease_risk）', fontsize=12)
axes[1, 0].set_ylabel('样本数量', fontsize=12)
axes[1, 0].set_title('训练集与测试集类别分布', fontsize=14, fontweight='bold')
axes[1, 0].set_xticks(x_pos)
axes[1, 0].set_xticklabels([f'类别{c}' for c in classes])
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(True, alpha=0.3, axis='y')
# 在柱子上标注数量
for i, (train_cnt, test_cnt) in enumerate(zip(train_class_count, test_class_count)):
    axes[1, 0].annotate(str(train_cnt), (i - width / 2, train_cnt), ha='center', va='bottom', fontsize=9)
    axes[1, 0].annotate(str(test_cnt), (i + width / 2, test_cnt), ha='center', va='bottom', fontsize=9)

# 子图4：混淆矩阵（查看模型对各类别的预测效果）
cm = confusion_matrix(y_test, all_preds)  # 计算混淆矩阵
# 绘制热力图
im = axes[1, 1].imshow(cm, cmap='Blues', aspect='auto')
# 添加数值标注
for i in range(len(classes)):
    for j in range(len(classes)):
        text = axes[1, 1].text(j, i, cm[i, j], ha='center', va='center', fontsize=10, fontweight='bold')
# 设置标签
axes[1, 1].set_xlabel('预测类别', fontsize=12)
axes[1, 1].set_ylabel('真实类别', fontsize=12)
axes[1, 1].set_title('测试集混淆矩阵', fontsize=14, fontweight='bold')
axes[1, 1].set_xticks(range(len(classes)))
axes[1, 1].set_yticks(range(len(classes)))
axes[1, 1].set_xticklabels([f'类别{c}' for c in classes])
axes[1, 1].set_yticklabels([f'类别{c}' for c in classes])
# 添加颜色条
cbar = plt.colorbar(im, ax=axes[1, 1])
cbar.set_label('样本数量', fontsize=10)

# 保存图片（高清格式）
plt.tight_layout()  # 自动调整子图间距
plt.savefig('health_model_analysis.png', dpi=300, bbox_inches='tight')  # dpi=300保证高清
plt.show()

# ---------------------- 6. 输出详细分类报告（补充文本版性能分析） ----------------------
print('=' * 60)
print('测试集分类性能详细报告')
print('=' * 60)
print(classification_report(
    y_test, all_preds,
    target_names=[f'疾病风险类别{c}' for c in classes],
    digits=4  # 保留4位小数
))
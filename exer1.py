import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 确保中文显示正常
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
df = pd.read_csv('train.csv')
print("数据集预览：")
print(df.head())

# 自动识别特征和目标列
# 假设最后一列是目标值
X_cols = df.columns[:-1]
y_col = df.columns[-1]

X = df[X_cols].values
y = df[y_col].values.reshape(-1, 1)

print(f"特征列: {X_cols.tolist()}, 目标列: {y_col}")

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 转换为Tensor
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)


# 定义线性模型
class LinearModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


model = LinearModel(X.shape[1])
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 记录训练过程
w_history = []
b_history = []
loss_history = []

# 训练模型
epochs = 1000
for epoch in range(epochs):
    y_pred = model(X_tensor)
    loss = criterion(y_pred, y_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 记录参数和损失
    w_history.append(model.linear.weight.detach().numpy().copy())
    b_history.append(model.linear.bias.item())
    loss_history.append(loss.item())

    # 打印训练进度
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}')

# 打印前几个和最后几个值检查
print("\n权重变化前几个值：")
print(w_history[:5])
print("\n权重变化最后几个值：")
print(w_history[-5:])
print("\n损失变化前几个值：")
print(loss_history[:5])
print("\n损失变化最后几个值：")
print(loss_history[-5:])

# 可视化w和loss的变化
plt.figure(figsize=(15, 5))

# w的变化曲线
plt.subplot(1, 3, 1)
w_np = np.array(w_history)
for i in range(w_np.shape[1]):
    plt.plot(w_np[:, i], label=f'w_{i + 1}')
plt.title('权重(w)训练过程变化')
plt.xlabel('Epoch')
plt.ylabel('w值')
plt.legend()

# loss的变化曲线
plt.subplot(1, 3, 2)
plt.plot(loss_history)
plt.title('损失(Loss)训练过程变化')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')  # 使用对数刻度更清晰

# w和loss的关系图
plt.subplot(1, 3, 3)
plt.scatter(w_np[:, 0], loss_history, alpha=0.6)
plt.title('w vs Loss')
plt.xlabel('w值')
plt.ylabel('Loss')
plt.grid(True)

plt.tight_layout()
plt.show()

# 可视化最终拟合结果（单特征情况）
if X.shape[1] == 1:
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, label='原始数据')
    plt.plot(X, model(X_tensor).detach().numpy(), color='red', label='拟合直线')
    plt.title('线性回归拟合结果')
    plt.xlabel(X_cols[0])
    plt.ylabel(y_col)
    plt.legend()
    plt.show()
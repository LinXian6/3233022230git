import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 读取数据
df = pd.read_csv('train.csv')
# 假设数据中有'x'和'y'两列
X = df['x'].values.reshape(-1, 1)
y = df['y'].values.reshape(-1, 1)

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 转换为Tensor
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)


# 定义线性模型
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # 输入维度1，输出维度1

    def forward(self, x):
        return self.linear(x)


model = LinearModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 记录训练过程
w_history = []
b_history = []
loss_history = []

# 训练模型
epochs = 1000
for epoch in range(epochs):
    # 前向传播
    y_pred = model(X_tensor)

    # 计算损失
    loss = criterion(y_pred, y_tensor)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 记录参数和损失
    w_history.append(model.linear.weight.item())
    b_history.append(model.linear.bias.item())
    loss_history.append(loss.item())

    # 打印训练进度
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# 可视化w和loss的变化
plt.figure(figsize=(12, 5))

# w的变化曲线
plt.subplot(1, 2, 1)
plt.plot(w_history)
plt.title('Weight (w) during training')
plt.xlabel('Epoch')
plt.ylabel('w value')

# loss的变化曲线
plt.subplot(1, 2, 2)
plt.plot(loss_history)
plt.title('Loss during training')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.tight_layout()
plt.show()

# 可视化最终拟合结果
plt.figure(figsize=(8, 6))
plt.scatter(X, y, label='Original data')
plt.plot(X, model(X_tensor).detach().numpy(), color='red', label='Fitted line')
plt.title('Linear Regression Result')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
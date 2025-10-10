import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

# 设置中文字体和图片清晰度
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 加载数据
data = pd.read_csv('train.csv')

# 2. 数据预处理
data = data.dropna(subset=['y'])  # 删除 y 缺失的行
Q1 = data['x'].quantile(0.25)
Q3 = data['x'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
data = data[(data['x'] >= lower_bound) & (data['x'] <= upper_bound)]

# 3. 转为张量
x = torch.tensor(data['x'].values, dtype=torch.float32).unsqueeze(1)
y = torch.tensor(data['y'].values, dtype=torch.float32).unsqueeze(1)

dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 4. 定义线性回归模型
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)
        # 正态分布初始化
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.linear.bias, mean=0.0, std=0.01)

    def forward(self, x):
        return self.linear(x)

# 5. 定义训练函数
def train(model, optimizer, criterion, epochs):
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
        epoch_loss /= len(dataloader.dataset)
        losses.append(epoch_loss)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}')
    return losses

# 6. 配置最佳超参数
model = LinearRegression()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 150

# 7. 开始训练
losses = train(model, optimizer, criterion, epochs)

# 8. 绘制损失曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs+1), losses, label='Adam (最佳模型)')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('最佳模型训练损失曲线')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# 9. 输出最终参数
print(f'\n训练完成！')
print(f'最终权重 w: {model.linear.weight.item():.4f}')
print(f'最终偏置 b: {model.linear.bias.item():.4f}')
print(f'最终损失: {losses[-1]:.4f}')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置 matplotlib 支持中文，使用 SimHei 字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取 CSV 文件
data = pd.read_csv('train.csv')

# 删除包含缺失值的行
data = data.dropna()

# 提取特征和目标变量
X = data['x'].values
y = data['y'].values

# 定义学习率和迭代次数
learning_rate = 0.000001
epochs = 100

# 重新初始化 w 和 b
w = 0
b = 0

# 存储不同 w 和 b 的损失值
w_losses = []
b_losses = []

# 存储 w 和 b 的值
w_values = []
b_values = []

# 设置图片清晰度
plt.rcParams['figure.dpi'] = 300

# 训练模型
for epoch in range(epochs):
    # 计算预测值
    y_pred = w * X + b

    # 计算损失
    loss = np.mean((y_pred - y) ** 2)

    # 计算梯度
    dw = np.mean(2 * (y_pred - y) * X)
    db = np.mean(2 * (y_pred - y))

    # 更新参数
    w = w - learning_rate * dw
    b = b - learning_rate * db

    # 存储 w 和 b 的值
    w_values.append(w)
    b_values.append(b)

    # 计算并存储当前 w 和 b 的损失值
    w_loss = np.mean(((w * X + np.mean(b_values)) - y) ** 2)
    b_loss = np.mean(((np.mean(w_values) * X + b) - y) ** 2)
    w_losses.append(w_loss)
    b_losses.append(b_loss)

# 绘制 w 和损失之间的关系图
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(w_values, w_losses)
plt.xlabel('w 值')
plt.xticks(rotation=45)
plt.ylabel('损失值')
plt.title('w 和损失之间的关系')

# 绘制 b 和损失之间的关系图
plt.subplot(1, 2, 2)
plt.plot(b_values, b_losses)
plt.xlabel('b 值')
plt.xticks(rotation=45)
plt.ylabel('损失值')
plt.title('b 和损失之间的关系')

plt.tight_layout()
# 保存图片
plt.savefig('result.png')
plt.show()

print(f"训练得到的 w: {w}")
print(f"训练得到的 b: {b}")
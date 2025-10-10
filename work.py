import torch
import pandas as pd
import matplotlib.pyplot as plt


# 方案1：直接指定支持负号的中文字体（Windows 系统优先）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans', 'Arial']  # 微软雅黑支持负号
plt.rcParams['axes.unicode_minus'] = True  # 保持开启，使用 Unicode 负号

data = pd.read_csv('train.csv')
data = data.dropna()

x_data = torch.Tensor(data['x'].values).reshape(-1, 1)
y_data = torch.Tensor(data['y'].values).reshape(-1, 1)


class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)
        # 初始化权重参数w和偏置参数b，使其满足正态分布
        torch.nn.init.normal_(self.linear.weight, mean=0.0, std=0.1)
        torch.nn.init.normal_(self.linear.bias, mean=0.0, std=0.1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


def train_with_optimizer(optimizer_name, lr=0.001, epochs=1000):
    """使用不同优化器训练模型"""
    model = LinearModel()
    criterion = torch.nn.MSELoss(reduction='mean')  # 使用mean避免数值过大

    # 为不同优化器设置合适的学习率
    if optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Adam通常需要更大学习率
    elif optimizer_name == 'Adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=0.1)  # Adagrad需要更大学习率

    losses = []
    weights = []
    biases = []

    for epoch in range(epochs):
        y_pred = model(x_data)
        loss = criterion(y_pred, y_data)

        # 检查NaN
        if torch.isnan(loss):
            print(f"{optimizer_name} - Epoch {epoch}: Loss is NaN")
            break

        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪防止爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        if epoch % 10 == 0:
            losses.append(loss.item())
            weights.append(model.linear.weight.item())
            biases.append(model.linear.bias.item())

    final_w = model.linear.weight.item()
    final_b = model.linear.bias.item()

    print(
        f"{optimizer_name} 最终参数: w={final_w:.6f}, b={final_b:.6f}, 最终损失: {losses[-1] if losses else 'NaN':.2f}")

    return losses, weights, biases, final_w, final_b


# 1. 比较不同优化器的性能
print("=" * 50)
print("不同优化器性能比较")
print("=" * 50)

optimizers = ['SGD', 'Adam', 'Adagrad']
results = {}

plt.figure(figsize=(15, 10))

for i, opt in enumerate(optimizers):
    print(f"\n训练 {opt} 优化器...")
    losses, weights, biases, final_w, final_b = train_with_optimizer(opt, epochs=1000)
    results[opt] = {
        'losses': losses,
        'weights': weights,
        'biases': biases,
        'final_w': final_w,
        'final_b': final_b
    }

    # 绘制损失曲线
    plt.subplot(2, 3, i + 1)
    plt.plot(losses)
    plt.title(f'{opt}优化器 - 损失曲线')
    plt.xlabel('训练轮次 (x10)')
    plt.ylabel('损失值')
    plt.grid(True)

# 2. 参数调节过程可视化
print("\n" + "=" * 50)
print("参数调节过程可视化")
print("=" * 50)

# 使用SGD进行详细参数调节可视化
model = LinearModel()
criterion = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

weights_history = []
biases_history = []
losses_history = []

print("初始化参数:")
print(f"w (初始): {model.linear.weight.item():.6f}")
print(f"b (初始): {model.linear.bias.item():.6f}")

for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    if epoch % 10 == 0:
        weights_history.append(model.linear.weight.item())
        biases_history.append(model.linear.bias.item())
        losses_history.append(loss.item())

# 参数w和b的调节过程
plt.subplot(2, 3, 4)
plt.plot(weights_history, label='权重 w')
plt.plot(biases_history, label='偏置 b')
plt.title('参数 w 和 b 的调节过程')
plt.xlabel('训练轮次 (x10)')
plt.ylabel('参数值')
plt.legend()
plt.grid(True)

# 参数空间轨迹
plt.subplot(2, 3, 5)
plt.plot(weights_history, biases_history, 'b-', alpha=0.7)
plt.scatter(weights_history[0], biases_history[0], color='red', label='起始点', s=50)
plt.scatter(weights_history[-1], biases_history[-1], color='green', label='最终点', s=50)
plt.title('参数空间轨迹 (w-b)')
plt.xlabel('权重 w')
plt.ylabel('偏置 b')
plt.legend()
plt.grid(True)

# 3. 学习率和epoch调节可视化
print("\n" + "=" * 50)
print("学习率和epoch调节可视化")
print("=" * 50)


def train_with_hyperparams(lr, epochs):
    """使用不同超参数训练模型"""
    model = LinearModel()
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    final_loss = float('inf')
    for epoch in range(epochs):
        y_pred = model(x_data)
        loss = criterion(y_pred, y_data)

        if torch.isnan(loss):
            break

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if epoch == epochs - 1:
            final_loss = loss.item()

    return final_loss


# 测试不同学习率（使用更安全的范围）
learning_rates = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
lr_losses = []
for lr in learning_rates:
    loss = train_with_hyperparams(lr, 500)
    lr_losses.append(loss)
    print(f"学习率 {lr:.2e}: 最终损失 = {loss:.2f}")

# 测试不同epoch数量
epochs_list = [100, 200, 500, 1000, 2000]
epoch_losses = []
for epochs in epochs_list:
    loss = train_with_hyperparams(0.001, epochs)
    epoch_losses.append(loss)
    print(f"训练轮次 {epochs}: 最终损失 = {loss:.2f}")

# 学习率影响可视化
plt.subplot(2, 3, 6)
plt.semilogx(learning_rates, lr_losses, 'ro-', linewidth=2, markersize=6)
plt.title('学习率对最终损失的影响')
plt.xlabel('学习率 (对数尺度)')
plt.ylabel('最终损失')
plt.grid(True)

plt.tight_layout()
plt.savefig('optimization_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

# 单独绘制epoch影响图
plt.figure(figsize=(10, 6))
plt.plot(epochs_list, epoch_losses, 'go-', linewidth=2, markersize=8)
plt.title('训练轮次对最终损失的影响')
plt.xlabel('训练轮次数量')
plt.ylabel('最终损失')
plt.grid(True)
plt.savefig('epoch_impact.png', dpi=300, bbox_inches='tight')
plt.show()

# 最终结果汇总
print("\n" + "=" * 50)
print("最终结果汇总")
print("=" * 50)
for opt in optimizers:
    result = results[opt]
    print(f"{opt}: w = {result['final_w']:.6f}, b = {result['final_b']:.6f}")

# 使用最佳参数进行最终预测
print("\n" + "=" * 50)
print("最终预测")
print("=" * 50)

final_model = LinearModel()
criterion = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.SGD(final_model.parameters(), lr=0.001)

print(f"初始参数: w={final_model.linear.weight.item():.6f}, b={final_model.linear.bias.item():.6f}")

for epoch in range(1000):
    y_pred = final_model(x_data)
    loss = criterion(y_pred, y_data)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=1.0)
    optimizer.step()

print(f"最终参数: w={final_model.linear.weight.item():.6f}, b={final_model.linear.bias.item():.6f}")

x_test = torch.Tensor([[4.0]])
y_test = final_model(x_test)
print(f"预测结果 x=4.0: y_pred={y_test.data.item():.6f}")
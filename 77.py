import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False


class LinearRegressionModel(nn.Module):
    """线性回归模型 y = wx + b"""

    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # 输入维度1，输出维度1

        # 初始化权重和偏置为正态分布
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.linear.bias, mean=0.0, std=0.1)

    def forward(self, x):
        return self.linear(x)


def train_model(x, y, optimizer_type='sgd', learning_rate=0.01, epochs=1000):
    """训练模型并返回历史记录，支持多种优化器"""
    # 转换为张量
    x_tensor = torch.FloatTensor(x).view(-1, 1)
    y_tensor = torch.FloatTensor(y).view(-1, 1)

    # 标准化数据
    x_mean, x_std = torch.mean(x_tensor), torch.std(x_tensor)
    y_mean, y_std = torch.mean(y_tensor), torch.std(y_tensor)
    x_normalized = (x_tensor - x_mean) / x_std
    y_normalized = (y_tensor - y_mean) / y_std

    # 初始化模型和损失函数
    model = LinearRegressionModel()
    criterion = nn.MSELoss()

    # 选择优化器
    if optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'adamax':
        optimizer = optim.Adamax(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'asgd':
        optimizer = optim.ASGD(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'rprop':
        optimizer = optim.Rprop(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'lbfgs':
        # LBFGS需要特殊处理，设置更多参数
        optimizer = optim.LBFGS(model.parameters(), lr=learning_rate,
                                history_size=100, max_iter=4)
    else:
        raise ValueError(f"不支持的优化器类型: {optimizer_type}")

    # 记录训练历史
    loss_history = []
    w_history = []
    b_history = []

    # 训练模型
    for epoch in range(epochs):
        # 对于LBFGS优化器需要特殊的closure函数
        if optimizer_type == 'lbfgs':
            def closure():
                optimizer.zero_grad()
                y_pred = model(x_normalized)
                loss = criterion(y_pred, y_normalized)
                loss.backward()
                return loss

            loss = optimizer.step(closure)
            loss_val = loss.item()
        else:
            # 前向传播
            y_pred = model(x_normalized)
            loss = criterion(y_pred, y_normalized)
            loss_val = loss.item()

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 记录当前状态
        loss_history.append(loss_val)
        w = model.linear.weight.item()
        b = model.linear.bias.item()
        w_history.append(w)
        b_history.append(b)

        # 每100轮打印信息
        if (epoch + 1) % 100 == 0:
            print(f"{optimizer_type} - 轮次 {epoch + 1}/{epochs}, 损失: {loss_val:.6f}, w: {w:.6f}, b: {b:.6f}")

    # 转换回原始尺度参数
    w_original = w * (y_std / x_std).item()
    b_original = (b * y_std + y_mean - w * (y_std / x_std) * x_mean).item()

    return model, w_original, b_original, loss_history, w_history, b_history


def compare_optimizers(x, y, optimizers, learning_rate=0.01, epochs=1000):
    """比较多种优化器的性能"""
    results = {}

    # 训练每种优化器
    for opt in optimizers:
        print(f"\n===== 开始使用 {opt} 优化器训练 =====")
        model, w, b, loss_hist, w_hist, b_hist = train_model(
            x, y, optimizer_type=opt, learning_rate=learning_rate, epochs=epochs)
        results[opt] = {
            'model': model,
            'w': w,
            'b': b,
            'loss_history': loss_hist,
            'w_history': w_hist,
            'b_history': b_hist
        }

    # 可视化比较结果
    plot_comparison_results(x, y, results, epochs)

    return results


def plot_comparison_results(x, y, results, epochs):
    """可视化不同优化器的性能比较"""
    optimizers = list(results.keys())
    num_optimizers = len(optimizers)

    # 创建画布
    plt.figure(figsize=(20, 16))

    # 1. 所有优化器的损失曲线对比
    plt.subplot(2, 2, 1)
    for opt in optimizers:
        plt.plot(range(epochs), results[opt]['loss_history'], label=opt, linewidth=1.5)
    plt.xlabel('训练轮次')
    plt.ylabel('损失值')
    plt.title('不同优化器的损失曲线对比')
    plt.legend()
    plt.grid(alpha=0.3)

    # 2. 所有优化器的权重w变化对比
    plt.subplot(2, 2, 2)
    for opt in optimizers:
        plt.plot(range(epochs), results[opt]['w_history'], label=opt, linewidth=1.5)
    plt.xlabel('训练轮次')
    plt.ylabel('权重 w')
    plt.title('不同优化器的权重w变化对比')
    plt.legend()
    plt.grid(alpha=0.3)

    # 3. 所有优化器的偏置b变化对比
    plt.subplot(2, 2, 3)
    for opt in optimizers:
        plt.plot(range(epochs), results[opt]['b_history'], label=opt, linewidth=1.5)
    plt.xlabel('训练轮次')
    plt.ylabel('偏置 b')
    plt.title('不同优化器的偏置b变化对比')
    plt.legend()
    plt.grid(alpha=0.3)

    # 4. 所有优化器的拟合结果对比
    plt.subplot(2, 2, 4)
    plt.scatter(x, y, color='skyblue', label='原始数据', alpha=0.6)
    for opt in optimizers:
        w = results[opt]['w']
        b = results[opt]['b']
        plt.plot(x, w * x + b, linewidth=2, label=f'{opt}: y = {w:.2f}x + {b:.2f}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('不同优化器的拟合结果对比')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 单独绘制每个优化器的详细参数变化图
    for opt in optimizers:
        plt.figure(figsize=(15, 10))

        # 子图1：损失曲线
        plt.subplot(2, 2, 1)
        plt.plot(range(epochs), results[opt]['loss_history'], color='darkorange', linewidth=1.5)
        plt.xlabel('训练轮次')
        plt.ylabel('损失值')
        plt.title(f'{opt} 损失值随训练轮次变化')
        plt.grid(alpha=0.3)

        # 子图2：w与loss关系
        plt.subplot(2, 2, 2)
        plt.plot(results[opt]['w_history'], results[opt]['loss_history'], color='forestgreen', linewidth=1.5)
        plt.xlabel('权重 w')
        plt.ylabel('损失值')
        plt.title(f'{opt} 权重 w 与损失的关系')
        plt.grid(alpha=0.3)

        # 子图3：b与loss关系
        plt.subplot(2, 2, 3)
        plt.plot(results[opt]['b_history'], results[opt]['loss_history'], color='darkorchid', linewidth=1.5)
        plt.xlabel('偏置 b')
        plt.ylabel('损失值')
        plt.title(f'{opt} 偏置 b 与损失的关系')
        plt.grid(alpha=0.3)

        # 子图4：拟合结果
        plt.subplot(2, 2, 4)
        plt.scatter(x, y, color='skyblue', label='原始数据', alpha=0.6)
        w = results[opt]['w']
        b = results[opt]['b']
        plt.plot(x, w * x + b, color='crimson', linewidth=2, label=f'拟合直线: y = {w:.2f}x + {b:.2f}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'{opt} 拟合结果')
        plt.legend()
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()


def parameter_tuning_visualization(x, y):
    """参数调节过程可视化：学习率和轮次"""
    optimizers = ['sgd', 'adam', 'rmsprop']  # 选择三种代表性优化器
    learning_rates = [0.001, 0.01, 0.1, 0.2]  # 不同学习率
    epochs_list = [500, 1000, 2000]  # 不同轮次

    # 学习率对模型性能的影响
    plt.figure(figsize=(15, 10))
    for i, opt in enumerate(optimizers, 1):
        plt.subplot(1, len(optimizers), i)
        for lr in learning_rates:
            _, _, _, loss_hist, _, _ = train_model(x, y, opt, learning_rate=lr, epochs=1000)
            plt.plot(range(1000), loss_hist, label=f'学习率={lr}', linewidth=1.5)
        plt.xlabel('训练轮次')
        plt.ylabel('损失值')
        plt.title(f'{opt} 不同学习率的损失曲线')
        plt.legend()
        plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 轮次对模型性能的影响
    plt.figure(figsize=(15, 10))
    for i, opt in enumerate(optimizers, 1):
        plt.subplot(1, len(optimizers), i)
        for epochs in epochs_list:
            _, _, _, loss_hist, _, _ = train_model(x, y, opt, learning_rate=0.01, epochs=epochs)
            plt.plot(range(epochs), loss_hist, label=f'轮次={epochs}', linewidth=1.5)
        plt.xlabel('训练轮次')
        plt.ylabel('损失值')
        plt.title(f'{opt} 不同轮次的损失曲线')
        plt.legend()
        plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    # 读取数据（尝试从当前目录查找train.csv）
    csv_path = r"D:\只想睡觉\下载\train.csv" # 修改为群里发的train.csv路径

    # 如果当前目录找不到，尝试其他常见路径
    if not os.path.exists(csv_path):
        # 尝试用户主目录
        home_dir = os.path.expanduser("~")
        csv_path = os.path.join(home_dir, "train.csv")
        if not os.path.exists(csv_path):
            # 尝试桌面
            desktop = os.path.join(home_dir, "Desktop", "train.csv")
            if os.path.exists(desktop):
                csv_path = desktop
            else:
                print(f"❌ 错误：未找到文件 train.csv")
                print(f"当前代码运行路径：{os.getcwd()}")
                return

    try:
        df = pd.read_csv(csv_path)
        print(f"✅ 成功读取数据：{csv_path}，共 {len(df)} 条记录")
        print(f"CSV文件列名：{df.columns.tolist()}")

        required_columns = ['x', 'y']
        # 尝试自动检测类似的列名
        if not all(col in df.columns for col in required_columns):
            print(f"⚠️ 警告：CSV未找到列 {required_columns}，尝试自动匹配...")
            # 寻找最相似的列名
            for col in required_columns:
                found = False
                for df_col in df.columns:
                    if col in df_col.lower():
                        required_columns[required_columns.index(col)] = df_col
                        found = True
                        print(f"  自动匹配：{col} -> {df_col}")
                if not found:
                    print(f"❌ 错误：未找到与 {col} 相似的列")
                    return

        df = df[required_columns].dropna()
        x = df[required_columns[0]].values
        y = df[required_columns[1]].values

        if len(x) == 0 or len(y) == 0:
            print("❌ 错误：x或y列中没有有效数据")
            return

        print(f"数据范围 - x: [{np.min(x):.2f}, {np.max(x):.2f}], y: [{np.min(y):.2f}, {np.max(y):.2f}]")

    except Exception as e:
        print(f"❌ 读取数据失败：{str(e)}")
        return

    # 定义要比较的优化器列表
    optimizers_to_compare = [
        'sgd', 'adagrad', 'adam',
        'adamax', 'asgd', 'rmsprop',
        'rprop', 'lbfgs'
    ]

    # 比较所有优化器
    print("\n📌 开始比较所有优化器...")
    results = compare_optimizers(x, y, optimizers_to_compare, learning_rate=0.01, epochs=1000)

    # 找到性能最好的优化器（损失最小）
    best_optimizer = min(optimizers_to_compare,
                         key=lambda opt: results[opt]['loss_history'][-1])
    print(f"\n🏆 性能最好的优化器是：{best_optimizer}，最终损失：{results[best_optimizer]['loss_history'][-1]:.6f}")

    # 参数调节可视化
    print("\n📊 开始参数调节可视化...")
    parameter_tuning_visualization(x, y)


if __name__ == "__main__":
    main()

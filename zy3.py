import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# 设置中文显示和字体，解决负号显示问题
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Arial Unicode MS"]
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

try:
    plt.switch_backend('TkAgg')
except Exception:
    pass


def load_data():
    """加载数据，优先使用train.csv，否则使用示例数据"""
    try:
        df = pd.read_csv('train.csv')
        required_cols = ['x', 'y']

        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            raise ValueError(f"train.csv缺少必要的列：{missing}")

        df_clean = df[required_cols].dropna()
        x_data = df_clean['x'].astype(float).values
        y_data = df_clean['y'].astype(float).values

        if len(x_data) < 2:
            raise ValueError(f"有效样本太少（{len(x_data)}个），至少需要2个样本")

        if np.std(x_data) < 1e-6:
            raise ValueError("x数据的标准差接近零，无法进行线性回归")

        print(f"成功读取train.csv，共{len(x_data)}个有效样本")
        print(f"x数据范围: [{x_data.min():.2f}, {x_data.max():.2f}]")
        print(f"y数据范围: [{y_data.min():.2f}, {y_data.max():.2f}]")
        return x_data, y_data

    except FileNotFoundError:
        print("未找到train.csv，使用示例数据演示")
        x_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        y_data = np.array([2.1, 4.0, 5.9, 8.2, 10.1, 11.9, 14.2, 15.8, 18.1, 20.3])
        return x_data, y_data

    except Exception as e:
        print(f"数据读取错误：{str(e)}，使用示例数据")
        x_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_data = np.array([2.1, 4.0, 5.8, 8.3, 10.2])
        return x_data, y_data


class LinearModel(nn.Module):
    """线性模型 y = w*x + b"""

    def __init__(self, mean=0.0, std=0.1):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # 输入维度1，输出维度1

        # 初始化权重和偏置为正态分布
        nn.init.normal_(self.linear.weight, mean=mean, std=std)
        nn.init.normal_(self.linear.bias, mean=mean, std=std)

    def forward(self, x):
        return self.linear(x)


def get_optimizer(model, optimizer_name, lr=0.005):
    """根据名称获取优化器"""
    optimizers = {
        'SGD': optim.SGD(model.parameters(), lr=lr),
        'Adam': optim.Adam(model.parameters(), lr=lr),
        'RMSprop': optim.RMSprop(model.parameters(), lr=lr),
        'Adagrad': optim.Adagrad(model.parameters(), lr=lr),
        'Adamax': optim.Adamax(model.parameters(), lr=lr)
    }
    return optimizers.get(optimizer_name, optim.SGD(model.parameters(), lr=lr))


def train_model(x_data, y_data, optimizer_name='SGD', epochs=300, lr=0.005):
    """使用PyTorch训练线性模型"""
    # 数据标准化
    x_mean, x_std = np.mean(x_data), np.std(x_data)
    y_mean, y_std = np.mean(y_data), np.std(y_data)

    x_scaled = (x_data - x_mean) / x_std if x_std > 1e-6 else x_data
    y_scaled = (y_data - y_mean) / y_std if y_std > 1e-6 else y_data

    # 转换为Tensor并添加维度
    x_tensor = torch.FloatTensor(x_scaled).unsqueeze(1)
    y_tensor = torch.FloatTensor(y_scaled).unsqueeze(1)

    # 创建数据集和数据加载器
    dataset = TensorDataset(x_tensor, y_tensor)
    batch_size = min(8, len(x_data))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型、损失函数和优化器
    model = LinearModel()
    criterion = nn.MSELoss()
    optimizer = get_optimizer(model, optimizer_name, lr)

    # 记录训练过程
    history = {
        'w': [],
        'b': [],
        'loss': [],
        'epoch': []
    }

    print(f"\n开始训练（{optimizer_name}）...")
    print(f"迭代次数: {epochs}, 学习率: {lr}, 批处理大小: {batch_size}")

    best_loss = float('inf')
    best_model = None

    for epoch in range(epochs):
        epoch_loss = 0.0

        for batch_x, batch_y in dataloader:
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_x.size(0)

        avg_loss = epoch_loss / len(x_data)

        # 恢复原始尺度的参数
        w_scaled = model.linear.weight.item()
        b_scaled = model.linear.bias.item()

        if y_std > 1e-6 and x_std > 1e-6:
            w = w_scaled * (y_std / x_std)
            b = b_scaled * y_std + y_mean - w * x_mean
        else:
            w = w_scaled
            b = b_scaled

        # 记录每一步的历史，而不仅仅是每5步
        history['w'].append(w)
        history['b'].append(b)
        history['loss'].append(avg_loss)
        history['epoch'].append(epoch)

        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], w: {w:.4f}, b: {b:.4f}, Loss: {avg_loss:.6f}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model = LinearModel()
            best_model.load_state_dict(model.state_dict())

    print(f"{optimizer_name} 训练完成!")
    print(f"最佳参数: w = {w:.4f}, b = {b:.4f}, 最小损失: {best_loss:.6f}")
    return best_model, history, (x_mean, x_std, y_mean, y_std)


def plot_optimizer_comparison(histories, optimizer_names, x_data, y_data, scalers):
    """比较不同优化器的性能"""
    x_mean, x_std, y_mean, y_std = scalers

    # 创建一个包含多个子图的图表
    fig = plt.figure(figsize=(18, 12))

    # 1. 损失对比图
    ax1 = fig.add_subplot(2, 2, 1)
    for i, history in enumerate(histories):
        ax1.plot(history['epoch'], history['loss'], label=optimizer_names[i], linewidth=2)
    ax1.set_title('不同优化器的损失变化', fontsize=14)
    ax1.set_xlabel('迭代次数', fontsize=12)
    ax1.set_ylabel('MSE损失', fontsize=12)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.7)
    ax1.legend()

    # 2. w参数对比图
    ax2 = fig.add_subplot(2, 2, 2)
    for i, history in enumerate(histories):
        ax2.plot(history['epoch'], history['w'], label=optimizer_names[i], linewidth=2)
    ax2.set_title('不同优化器的w参数变化', fontsize=14)
    ax2.set_xlabel('迭代次数', fontsize=12)
    ax2.set_ylabel('w值', fontsize=12)
    ax2.grid(True, alpha=0.7)
    ax2.legend()

    # 3. b参数对比图
    ax3 = fig.add_subplot(2, 2, 3)
    for i, history in enumerate(histories):
        ax3.plot(history['epoch'], history['b'], label=optimizer_names[i], linewidth=2)
    ax3.set_title('不同优化器的b参数变化', fontsize=14)
    ax3.set_xlabel('迭代次数', fontsize=12)
    ax3.set_ylabel('b值', fontsize=12)
    ax3.grid(True, alpha=0.7)
    ax3.legend()

    # 4. 拟合结果对比
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.scatter(x_data, y_data, c='blue', alpha=0.7, label='原始数据')

    x_range = np.linspace(x_data.min(), x_data.max(), 100)
    x_scaled = (x_range - x_mean) / x_std if x_std > 1e-6 else x_range
    x_tensor = torch.FloatTensor(x_scaled).unsqueeze(1)

    for i, (model, history) in enumerate(zip(models, histories)):
        with torch.no_grad():
            y_pred_scaled = model(x_tensor).numpy().flatten()
            if y_std > 1e-6:
                y_pred = y_pred_scaled * y_std + y_mean
            else:
                y_pred = y_pred_scaled
        ax4.plot(x_range, y_pred, linewidth=2, label=f'{optimizer_names[i]}拟合')

    ax4.set_title('不同优化器的拟合结果', fontsize=14)
    ax4.set_xlabel('x', fontsize=12)
    ax4.set_ylabel('y', fontsize=12)
    ax4.grid(True, alpha=0.7)
    ax4.legend()

    plt.tight_layout()
    plt.savefig('optimizer_comparison.png', dpi=300, bbox_inches='tight')
    print("优化器对比图已保存为：optimizer_comparison.png")
    plt.show()


def plot_parameter_tuning(history, optimizer_name):
    """可视化参数w和b的调节过程"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'{optimizer_name}优化器的参数调节过程', fontsize=14)

    # w参数调节过程
    ax1.plot(history['epoch'], history['w'], 'b-', linewidth=2)
    ax1.set_title('权重w的调节过程', fontsize=12)
    ax1.set_xlabel('迭代次数', fontsize=11)
    ax1.set_ylabel('w值', fontsize=11)
    ax1.grid(True, alpha=0.7)

    # b参数调节过程
    ax2.plot(history['epoch'], history['b'], 'g-', linewidth=2)
    ax2.set_title('偏置b的调节过程', fontsize=12)
    ax2.set_xlabel('迭代次数', fontsize=11)
    ax2.set_ylabel('b值', fontsize=11)
    ax2.grid(True, alpha=0.7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'{optimizer_name}_parameter_tuning.png', dpi=300, bbox_inches='tight')
    print(f"{optimizer_name}参数调节过程图已保存为：{optimizer_name}_parameter_tuning.png")
    plt.show()


def plot_hyperparameter_effects(x_data, y_data, scalers):
    """可视化不同超参数（epoch和学习率）的影响"""
    x_mean, x_std, y_mean, y_std = scalers

    # 测试不同的学习率
    learning_rates = [0.001, 0.005, 0.01, 0.05]
    lr_histories = []

    print("\n测试不同学习率的影响...")
    for lr in learning_rates:
        _, history, _ = train_model(x_data, y_data, optimizer_name='Adam', epochs=200, lr=lr)
        lr_histories.append(history)

    # 测试不同的迭代次数
    epochs_list = [50, 100, 200, 400]
    epoch_histories = []

    print("\n测试不同迭代次数的影响...")
    for epochs in epochs_list:
        _, history, _ = train_model(x_data, y_data, optimizer_name='Adam', epochs=epochs, lr=0.005)
        epoch_histories.append(history)

    # 创建可视化图表
    fig = plt.figure(figsize=(18, 10))

    # 1. 不同学习率的损失对比
    ax1 = fig.add_subplot(2, 2, 1)
    for i, history in enumerate(lr_histories):
        ax1.plot(history['epoch'], history['loss'], label=f'学习率={learning_rates[i]}', linewidth=2)
    ax1.set_title('不同学习率对损失的影响', fontsize=14)
    ax1.set_xlabel('迭代次数', fontsize=12)
    ax1.set_ylabel('MSE损失', fontsize=12)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.7)
    ax1.legend()

    # 2. 不同迭代次数的损失对比
    ax2 = fig.add_subplot(2, 2, 2)
    for i, history in enumerate(epoch_histories):
        ax2.plot(history['epoch'], history['loss'], label=f'迭代次数={epochs_list[i]}', linewidth=2)
    ax2.set_title('不同迭代次数对损失的影响', fontsize=14)
    ax2.set_xlabel('迭代次数', fontsize=12)
    ax2.set_ylabel('MSE损失', fontsize=12)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.7)
    ax2.legend()

    # 3. 不同学习率的w参数对比
    ax3 = fig.add_subplot(2, 2, 3)
    for i, history in enumerate(lr_histories):
        ax3.plot(history['epoch'], history['w'], label=f'学习率={learning_rates[i]}', linewidth=2)
    ax3.set_title('不同学习率对w参数的影响', fontsize=14)
    ax3.set_xlabel('迭代次数', fontsize=12)
    ax3.set_ylabel('w值', fontsize=12)
    ax3.grid(True, alpha=0.7)
    ax3.legend()

    # 4. 不同迭代次数的w参数对比
    ax4 = fig.add_subplot(2, 2, 4)
    for i, history in enumerate(epoch_histories):
        ax4.plot(history['epoch'], history['w'], label=f'迭代次数={epochs_list[i]}', linewidth=2)
    ax4.set_title('不同迭代次数对w参数的影响', fontsize=14)
    ax4.set_xlabel('迭代次数', fontsize=12)
    ax4.set_ylabel('w值', fontsize=12)
    ax4.grid(True, alpha=0.7)
    ax4.legend()

    plt.tight_layout()
    plt.savefig('hyperparameter_effects.png', dpi=300, bbox_inches='tight')
    print("超参数影响图已保存为：hyperparameter_effects.png")
    plt.show()


if __name__ == "__main__":
    try:
        # 加载数据
        x_data, y_data = load_data()

        # 选择三种优化器进行比较
        optimizers_to_test = ['SGD', 'Adam', 'RMSprop']
        models = []
        histories = []
        scalers = None

        # 使用每种优化器训练模型
        for opt_name in optimizers_to_test:
            model, history, scalers = train_model(
                x_data,
                y_data,
                optimizer_name=opt_name,
                epochs=300,
                lr=0.005
            )
            models.append(model)
            histories.append(history)

            # 可视化每种优化器的参数调节过程
            plot_parameter_tuning(history, opt_name)

        # 比较不同优化器的性能
        plot_optimizer_comparison(histories, optimizers_to_test, x_data, y_data, scalers)

        # 可视化超参数(学习率和迭代次数)的影响
        plot_hyperparameter_effects(x_data, y_data, scalers)

    except Exception as e:
        print(f"程序运行出错: {str(e)}")

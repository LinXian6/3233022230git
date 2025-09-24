import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

try:
    plt.switch_backend('TkAgg')
except Exception:
    pass


def load_data():
    try:
        df = pd.read_csv('train.csv')
        required_cols = ['x', 'y']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"train.csv必须包含列：{required_cols}")

        df = df[required_cols].dropna()
        x_data = df['x'].astype(float).values
        y_data = df['y'].astype(float).values

        if len(x_data) == 0:
            raise ValueError("数据清理后无有效样本")

        print(f"成功读取train.csv，共{len(x_data)}个有效样本")
        return x_data, y_data

    except FileNotFoundError:
        print("未找到train.csv，使用示例数据演示")
        x_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_data = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        return x_data, y_data

    except Exception as e:
        print(f"数据读取错误：{str(e)}，使用示例数据")
        x_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_data = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        return x_data, y_data


def forward(x, w, b):
    return w * x + b


def compute_mse(x_data, y_data, w, b):
    y_pred = forward(x_data, w, b)
    return np.mean((y_pred - y_data) ** 2)


def analyze_param_loss(x_data, y_data):
    x_range = x_data.max() - x_data.min()
    y_range = y_data.max() - y_data.min()
    init_w = y_range / x_range if x_range != 0 else 2.0
    init_b = np.mean(y_data - init_w * x_data)

    fixed_b = init_b
    w_min, w_max = init_w - 3, init_w + 3
    w_values = np.arange(w_min, w_max, 0.1)
    loss_w = [compute_mse(x_data, y_data, w, fixed_b) for w in w_values]

    fixed_w = init_w
    b_min, b_max = init_b - 3, init_b + 3
    b_values = np.arange(b_min, b_max, 0.1)
    loss_b = [compute_mse(x_data, y_data, fixed_w, b) for b in b_values]

    return w_values, loss_w, b_values, loss_b, fixed_w, fixed_b


def plot_param_loss(w_values, loss_w, b_values, loss_b, fixed_w, fixed_b):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(w_values, loss_w, 'b-', linewidth=2, label='损失曲线')
    plt.axvline(x=fixed_w, color='r', linestyle='--', label=f'初始w={fixed_w:.2f}')
    plt.xlabel('权重 w', fontsize=11)
    plt.ylabel('均方误差（Loss）', fontsize=11)
    plt.title('w与损失值的关系', fontsize=12)
    plt.grid(True, alpha=0.7)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(b_values, loss_b, 'g-', linewidth=2, label='损失曲线')
    plt.axvline(x=fixed_b, color='r', linestyle='--', label=f'初始b={fixed_b:.2f}')
    plt.xlabel('偏置 b', fontsize=11)
    plt.ylabel('均方误差（Loss）', fontsize=11)
    plt.title('b与损失值的关系', fontsize=12)
    plt.grid(True, alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plt.savefig('param_loss_relation.png', dpi=300, bbox_inches='tight')
    print("图形已保存为：param_loss_relation.png")
    plt.show()
    plt.pause(10)


def train_best_param(x_data, y_data, fixed_w, fixed_b):
    w_search = np.arange(fixed_w - 1, fixed_w + 1, 0.01)
    b_search = np.arange(fixed_b - 1, fixed_b + 1, 0.01)

    best_loss = float('inf')
    best_w = fixed_w
    best_b = fixed_b

    for w in w_search:
        for b in b_search:
            current_loss = compute_mse(x_data, y_data, w, b)
            if current_loss < best_loss:
                best_loss = current_loss
                best_w = w
                best_b = b

    print("\n🏆 训练完成：最优参数")
    print(f"w = {best_w:.4f}, b = {best_b:.4f}")
    print(f"最小损失（MSE） = {best_loss:.6f}")
    return best_w, best_b, best_loss


if __name__ == "__main__":
    x_data, y_data = load_data()
    w_values, loss_w, b_values, loss_b, fixed_w, fixed_b = analyze_param_loss(x_data, y_data)
    plot_param_loss(w_values, loss_w, b_values, loss_b, fixed_w, fixed_b)
    best_w, best_b, best_loss = train_best_param(x_data, y_data, fixed_w, fixed_b)
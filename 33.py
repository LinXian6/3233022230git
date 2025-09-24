import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False


class LinearRegression:
    """线性回归模型 y = wx + b"""

    def __init__(self, learning_rate=0.001, epochs=1000):  # 调整了默认学习率
        self.w = np.random.randn()  # 随机初始化权重
        self.b = np.random.randn()  # 随机初始化偏置
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.loss_history = []  # 损失历史
        self.w_history = []  # 权重历史
        self.b_history = []  # 偏置历史

    def compute_loss(self, x, y):
        """计算均方误差损失"""
        n = len(y)
        y_pred = self.w * x + self.b
        loss = np.sum((y_pred - y) ** 2) / n
        return loss

    def gradient_descent(self, x, y):
        """梯度下降更新参数"""
        n = len(y)
        y_pred = self.w * x + self.b
        # 计算梯度
        dw = (2 / n) * np.sum((y_pred - y) * x)
        db = (2 / n) * np.sum(y_pred - y)
        # 更新参数
        self.w -= self.learning_rate * dw
        self.b -= self.learning_rate * db

    def train(self, x, y):
        """训练模型"""
        # 标准化数据以提高训练稳定性
        x_mean, x_std = np.mean(x), np.std(x)
        y_mean, y_std = np.mean(y), np.std(y)
        x_normalized = (x - x_mean) / x_std
        y_normalized = (y - y_mean) / y_std

        for epoch in range(self.epochs):
            # 记录当前状态
            current_loss = self.compute_loss(x_normalized, y_normalized)
            self.loss_history.append(current_loss)
            self.w_history.append(self.w)
            self.b_history.append(self.b)

            # 梯度下降更新
            self.gradient_descent(x_normalized, y_normalized)

            # 每100轮打印信息
            if (epoch + 1) % 100 == 0:
                print(f"轮次 {epoch + 1}/{self.epochs}, 损失: {current_loss:.6f}, w: {self.w:.6f}, b: {self.b:.6f}")

        # 调试信息：打印历史记录长度
        print(f"\n训练过程记录：")
        print(f"损失记录长度: {len(self.loss_history)}")
        print(f"权重记录长度: {len(self.w_history)}")
        print(f"偏置记录长度: {len(self.b_history)}")

        # 转换回原始数据尺度的参数
        self.w_original = self.w * (y_std / x_std)
        self.b_original = (self.b * y_std) + y_mean - (self.w * (y_std / x_std) * x_mean)

        return self.w_original, self.b_original


def main():
    # 1. 读取数据
    # 请修改为您的实际文件路径
    csv_path = r"D:\只想睡觉\下载\train.csv"

    # 检查文件是否存在
    if not os.path.exists(csv_path):
        print(f"❌ 错误：未找到文件 {csv_path}")
        print(f"当前代码运行路径：{os.getcwd()}")
        return

    try:
        # 读取CSV
        df = pd.read_csv(csv_path)
        print(f"✅ 成功读取数据：{csv_path}，共 {len(df)} 条记录")
        print(f"CSV文件列名：{df.columns.tolist()}")

        # 验证必要列
        required_columns = ['x', 'y']
        if not all(col in df.columns for col in required_columns):
            print(f"❌ 错误：CSV需包含列 {required_columns}，当前列名是 {df.columns.tolist()}")
            return

        # 提取x和y并去除可能的缺失值
        df = df[required_columns].dropna()
        x = df['x'].values
        y = df['y'].values

        if len(x) == 0 or len(y) == 0:
            print("❌ 错误：x或y列中没有有效数据")
            return

        print(f"数据范围 - x: [{np.min(x):.2f}, {np.max(x):.2f}], y: [{np.min(y):.2f}, {np.max(y):.2f}]")

    except Exception as e:
        print(f"❌ 读取数据失败：{str(e)}")
        return

    # 2. 训练模型
    model = LinearRegression(learning_rate=0.01, epochs=1000)
    print("\n📌 开始训练模型...")
    w, b = model.train(x, y)
    print(f"\n✅ 训练完成！最终参数：w = {w:.6f}, b = {b:.6f}")

    # 3. 绘制结果
    plt.figure(figsize=(15, 12))

    # 子图1：数据点+拟合直线
    plt.subplot(2, 2, 1)
    plt.scatter(x, y, color='skyblue', label='原始数据', alpha=0.6)
    plt.plot(x, w * x + b, color='crimson', linewidth=2, label=f'拟合直线: y = {w:.2f}x + {b:.2f}')
    plt.xlabel('x', fontsize=10)
    plt.ylabel('y', fontsize=10)
    plt.title('数据分布与线性拟合', fontsize=12)
    plt.legend()

    # 子图2：w与loss关系
    plt.subplot(2, 2, 2)
    if len(model.w_history) > 0 and len(model.loss_history) > 0:
        plt.plot(model.w_history, model.loss_history, color='forestgreen', linewidth=1.5)
        plt.xlabel('权重 w', fontsize=10)
        plt.ylabel('损失值', fontsize=10)
        plt.title('权重 w 与损失的变化关系', fontsize=12)
    else:
        plt.text(0.5, 0.5, '无权重历史数据', ha='center', va='center', transform=plt.gca().transAxes)

    # 子图3：b与loss关系
    plt.subplot(2, 2, 3)
    if len(model.b_history) > 0 and len(model.loss_history) > 0:
        plt.plot(model.b_history, model.loss_history, color='darkorchid', linewidth=1.5)
        plt.xlabel('偏置 b', fontsize=10)
        plt.ylabel('损失值', fontsize=10)
        plt.title('偏置 b 与损失的变化关系', fontsize=12)
    else:
        plt.text(0.5, 0.5, '无偏置历史数据', ha='center', va='center', transform=plt.gca().transAxes)

    # 子图4：损失随轮次变化
    plt.subplot(2, 2, 4)
    if len(model.loss_history) > 0:
        plt.plot(range(len(model.loss_history)), model.loss_history, color='darkorange', linewidth=1.5)
        plt.xlabel('训练轮次', fontsize=10)
        plt.ylabel('损失值', fontsize=10)
        plt.title('损失值随训练轮次下降趋势', fontsize=12)
    else:
        plt.text(0.5, 0.5, '无损失历史数据', ha='center', va='center', transform=plt.gca().transAxes)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

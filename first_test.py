import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer


class StrokeDataset(Dataset):
    def __init__(self, filepath):
        # 读取CSV文件
        data = pd.read_csv(filepath)

        # 数据预处理
        # 1. 删除id列
        data = data.drop('id', axis=1)

        # 2. 处理分类变量
        categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            label_encoders[col] = le

        # 3. 处理缺失值
        numerical_cols = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']
        imputer = SimpleImputer(strategy='mean')
        data[numerical_cols] = imputer.fit_transform(data[numerical_cols])

        # 4. 特征标准化
        scaler = StandardScaler()
        feature_cols = [col for col in data.columns if col != 'stroke']
        data[feature_cols] = scaler.fit_transform(data[feature_cols])

        # 转换为numpy数组
        xy = data.astype(np.float32).values
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])  # 所有特征
        self.y_data = torch.from_numpy(xy[:, [-1]])  # 目标变量

        self.feature_dim = self.x_data.shape[1]  # 存储特征维度

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class Model(torch.nn.Module):
    def __init__(self, input_dim):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, 64)
        self.linear2 = torch.nn.Linear(64, 32)
        self.linear3 = torch.nn.Linear(32, 16)
        self.linear4 = torch.nn.Linear(16, 1)  # 输出logits，不要sigmoid
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.3)
        # 移除最后的sigmoid，因为BCEWithLogitsLoss内部包含sigmoid

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.relu(self.linear2(x))
        x = self.dropout(x)
        x = self.relu(self.linear3(x))
        x = self.linear4(x)  # 直接输出logits
        return x


def calculate_class_weights(dataset):
    """计算类别权重以处理不平衡数据"""
    labels = torch.cat([dataset[i][1] for i in range(len(dataset))])
    class_counts = torch.bincount(labels.long().flatten())
    print(f"类别分布: 没有中风-{class_counts[0].item()}, 中风-{class_counts[1].item()}")

    # 计算正样本权重
    pos_weight = torch.tensor([class_counts[0] / class_counts[1]])
    print(f"正样本权重: {pos_weight.item():.2f}")

    return pos_weight


if __name__ == '__main__':
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 创建数据集
    full_dataset = StrokeDataset('healthcare-dataset-stroke-data.csv')

    # 获取特征维度
    input_dim = full_dataset.feature_dim
    print(f"特征维度: {input_dim}")

    # 计算类别权重
    pos_weight = calculate_class_weights(full_dataset)

    # 数据集分割
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(
        full_dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # 数据加载器
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    # 模型初始化
    model = Model(input_dim=input_dim)

    # 使用带权重的BCEWithLogitsLoss
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005, weight_decay=1e-5)

    # 添加学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    # 训练记录
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    num_epochs = 100
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_running_loss = 0.0
        train_preds = []
        train_labels = []

        for inputs, labels in train_loader:
            # 确保数据格式正确
            inputs = inputs.float()
            labels = labels.float()

            # 前向传播
            logits = model(inputs)
            loss = criterion(logits, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_running_loss += loss.item() * inputs.size(0)

            # 预测
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                preds = torch.round(probs)
                train_preds.extend(preds.detach().numpy().flatten())
                train_labels.extend(labels.numpy().flatten())

        # 更新学习率
        scheduler.step()

        # 计算训练指标
        train_epoch_loss = train_running_loss / len(train_dataset)
        train_epoch_acc = accuracy_score(train_labels, train_preds)
        train_losses.append(train_epoch_loss)
        train_accs.append(train_epoch_acc)

        # 测试阶段
        model.eval()
        test_running_loss = 0.0
        test_preds = []
        test_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.float()
                labels = labels.float()

                logits = model(inputs)
                loss = criterion(logits, labels)

                test_running_loss += loss.item() * inputs.size(0)

                probs = torch.sigmoid(logits)
                preds = torch.round(probs)
                test_preds.extend(preds.numpy().flatten())
                test_labels.extend(labels.numpy().flatten())

        # 计算测试指标
        test_epoch_loss = test_running_loss / len(test_dataset)
        test_epoch_acc = accuracy_score(test_labels, test_preds)
        test_losses.append(test_epoch_loss)
        test_accs.append(test_epoch_acc)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}]')
            print(f'  训练准确率: {train_epoch_acc:.4f}, 训练损失: {train_epoch_loss:.6f}')
            print(f'  测试准确率: {test_epoch_acc:.4f}, 测试损失: {test_epoch_loss:.6f}\n')

    # 最终结果
    print('=' * 50)
    print('最终结果:')
    print(f'最终训练准确率: {train_accs[-1]:.4f}')
    print(f'最终测试准确率: {test_accs[-1]:.4f}')

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    # 可视化
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label='训练损失')
    plt.plot(range(1, num_epochs + 1), test_losses, label='测试损失')
    plt.title('损失 vs 训练轮次')
    plt.xlabel('训练轮次')
    plt.ylabel('损失')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accs, label='训练准确率')
    plt.plot(range(1, num_epochs + 1), test_accs, label='测试准确率')
    plt.title('准确率 vs 训练轮次')
    plt.xlabel('训练轮次')
    plt.ylabel('准确率')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # 保存模型
    model_save_path = 'stroke_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'pos_weight': pos_weight
    }, model_save_path)
    print(f'模型已保存至: {model_save_path}')
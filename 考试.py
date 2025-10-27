import pandas as pd
import numpy as np
import os  # 用于判断文件是否存在
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision
from tensorflow.keras.callbacks import Callback  # 导入回调基类
import matplotlib.pyplot as plt


# --------------------------
# 自定义回调：每轮训练后记录指标
# --------------------------
class EpochLogger(Callback):
    def __init__(self, log_file='训练指标记录.csv'):
        self.log_file = log_file
        # 定义需要记录的指标列名（与训练时的指标对应）
        self.columns = ['epoch', 'accuracy', 'precision', 'val_accuracy', 'val_precision', 'loss', 'val_loss']

    def on_train_begin(self, logs=None):
        # 训练开始时，若文件不存在则创建并写入表头
        if not os.path.exists(self.log_file):
            df = pd.DataFrame(columns=self.columns)
            df.to_csv(self.log_file, index=False)

    def on_epoch_end(self, epoch, logs=None):
        # 每轮结束后，提取当前轮次的指标（epoch从0开始，+1转为1-based）
        current_epoch = epoch + 1
        # 从logs中获取当前轮次的指标（logs包含训练/验证的loss和metrics）
        log_data = {
            'epoch': current_epoch,
            'accuracy': logs.get('accuracy'),
            'precision': logs.get('precision'),
            'val_accuracy': logs.get('val_accuracy'),
            'val_precision': logs.get('val_precision'),
            'loss': logs.get('loss'),
            'val_loss': logs.get('val_loss')
        }
        # 将当前轮次数据追加到CSV
        df = pd.DataFrame([log_data])
        df.to_csv(self.log_file, mode='a', header=False, index=False)
        print(f"\n第{current_epoch}轮指标已记录到 {self.log_file}")


# 1. 加载数据
data = pd.read_csv('D:/只想睡觉/下载/exam/data.csv')

# 2. 数据预处理
print("数据基本信息：")
data.info()

# 处理目标变量（0=良性B，1=恶性M）
le = LabelEncoder()
data['diagnosis'] = le.fit_transform(data['diagnosis'])
print("\n目标变量分布：")
print(data['diagnosis'].value_counts())

# 提取特征和目标变量
X = data.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1)
y = data['diagnosis']

# 划分训练集和测试集（保持类别平衡）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. 构建模型
model = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# 编译模型
model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy', Precision(name='precision')]
)

# 4. 训练模型（加入自定义回调，实现每轮记录）
print("\n开始训练模型...")
history = model.fit(
    X_train_scaled, y_train,
    epochs=80,
    batch_size=32,
    validation_split=0.2,
    verbose=1,
    callbacks=[EpochLogger()]  # 传入自定义回调实例
)

# 5. 预览记录的指标（可选）
history_df = pd.read_csv('训练指标记录.csv')
print("\n前5轮训练指标：")
print(history_df.head())
print("\n最后5轮训练指标：")
print(history_df.tail())

# 6. 可视化精准度变化
plt.figure(figsize=(10, 6))
plt.plot(history_df['epoch'], history_df['precision'], label='训练集精准度')
plt.plot(history_df['epoch'], history_df['val_precision'], label='验证集精准度')
plt.xlabel('训练轮次（epoch）')
plt.ylabel('精准度（Precision）')
plt.title('每次训练的精准度变化')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# 7. 测试集最终评估
y_pred_prob = model.predict(X_test_scaled)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()
print("\n测试集最终评估报告：")
print(classification_report(y_test, y_pred, target_names=['良性（B）', '恶性（M）']))
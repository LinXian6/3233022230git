import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

#
df = pd.read_csv("earthquake_alert_balanced_dataset.csv")

print("===== 数据基本信息 =====")
print(f"数据形状：{df.shape}")
print("\n缺失值统计：")
print(df.isnull().sum())
print("\nalert标签分布（多类别确认）：")
print(df["alert"].value_counts())

# 4.1 特征与标签分离
X = df.drop("alert", axis=1)
y = df["alert"]

# 4.2 处理缺失值
if df.isnull().sum().sum() > 0:
    X = X.fillna(X.mean())
    print("\n已处理缺失值（采用均值填充）")

le = LabelEncoder()
y_encoded = le.fit_transform(y)
n_classes = len(le.classes_)
print(f"\n标签编码映射：{dict(zip(le.classes_, range(n_classes)))}")

# 特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#  划分训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)
print(f"\n训练集规模：{X_train.shape}，测试集规模：{X_test.shape}")

# 5. 构建全连接神经网络
model = Sequential([
    Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(n_classes, activation="softmax")
])

print("\n===== 模型结构 =====")
model.summary()

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 6. 训练模型
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=100,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# 7. 模型评估
print("\n===== 测试集评估结果 =====")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"测试集准确率：{test_acc:.4f}")

# 生成分类报告
y_pred_proba = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)

print("\n===== 类别级分类报告 =====")
print(classification_report(
    y_test, y_pred,
    target_names=le.classes_,
    digits=4
))

# 中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(12, 5))

# 损失曲线
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="训练损失")
plt.plot(history.history["val_loss"], label="验证损失")
plt.title("训练损失 vs 验证损失")
plt.xlabel("轮次（Epochs）")
plt.ylabel("损失值")
plt.legend()
plt.grid(alpha=0.3)

# 准确率曲线
plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="训练准确率")
plt.plot(history.history["val_accuracy"], label="验证准确率")
plt.title("训练准确率 vs 验证准确率")
plt.xlabel("轮次（Epochs）")
plt.ylabel("准确率")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()
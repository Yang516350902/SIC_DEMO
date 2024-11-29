import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

"""
1. 模型架构（Architecture）
本次实验使用的是多层感知机（MLP）模型，是一种经典的前馈神经网络（Feedforward Neural Network），用于预测多变量回归任务。具体架构如下：

    (1) 输入层（Input Layer）：
    特征数：模型接收 9 个输入特征：
    Pressure (mT), SRF-C (W), SRF-E (W), BRF (W), SF6 (sccm), Ar (sccm), He Pressure (T), Chiller Temp (℃), Time (s)
    数据标准化：对所有输入特征进行了标准化处理（均值为 0，方差为 1），以减少特征尺度差异对模型训练的影响。
    (2) 隐藏层（Hidden Layers）：
    第一隐藏层：
    神经元数量：64 个
    激活函数：ReLU（Rectified Linear Unit），具有良好的非线性表示能力，能够有效缓解梯度消失问题。
    正则化：使用 Dropout（丢弃率 0.3），随机丢弃 30% 的神经元，以减少模型过拟合的风险。
    第二隐藏层：
    神经元数量：64 个
    激活函数：ReLU。
    正则化：使用 Dropout（丢弃率 0.3）。
    (3) 输出层（Output Layer）：
    输出神经元数量：2 个，分别对应两个目标变量：
    Etching Rate (nm/min)：刻蚀速率预测。
    Angle (°)：刻蚀角度预测。
    激活函数：无（线性激活），用于输出连续值。
2. 技术细节（Technical Details）
    (1) 损失函数（Loss Function）：
    均方误差（MSE）：用于回归任务的损失计算。计算预测值与实际值之间的平方差，并将其平均化。
    (2) 优化器（Optimizer）：
    Adam 优化器：一种自适应学习率优化算法，结合了 AdaGrad 和 RMSProp 的优点，能够加速模型收敛，并对学习率进行动态调整，稳定性较强。
    (3) 评估指标（Evaluation Metrics）：
    平均绝对误差（MAE）：用于评估模型预测值与实际值的平均偏差。
    (4) 正则化技术（Regularization Techniques）：
    Dropout：在每个隐藏层中使用 Dropout 层，以防止模型过拟合。在训练过程中随机丢弃一定比例的神经元（本实验中为 30%），减少对个别神经元的依赖。
3. 数据集划分与模型训练过程（Data Splitting and Model Training Process）
    (1) 数据集划分（Data Splitting）：
    总数据量：32 条。
    测试集（Test Set）：手动选择前 5 条数据（16%），用于最终模型评估，完全不参与训练和验证。
    训练和验证集（Training and Validation Sets）：其余 27 条数据（84%），用于交叉验证模型训练和性能评估。
    (2) 交叉验证（Cross-Validation）：
    方法：使用 5 折交叉验证（K-Fold Cross-Validation）。
    步骤：将 27 条数据分成 5 份，每次使用 4 份作为训练集（80%），1 份作为验证集（20%）。
    重复训练：模型在每一折训练时，都重新训练和验证，总共训练 5 次。
    评估：每次训练后，计算验证集的 MAE，并记录每一折的表现，最终计算 5 折交叉验证的平均 MAE。
    (3) 模型训练（Model Training）：
    训练集大小：每次折叠的训练集包含约 21-22 条数据。
    验证集大小：每次折叠的验证集包含约 5-6 条数据。
    训练轮次（Epochs）：100 轮。
    批次大小（Batch Size）：16。
    训练优化：在每个折叠的训练过程中，模型参数被不断更新，以最小化损失函数（MSE）。
    (4) 最终模型训练与保存（Final Model Training and Saving）：
    使用所有 27 条数据重新训练最终模型，并将其保存为 v1.h5 文件。
    同时保存标准化参数（均值和标准差）以便后续对新数据进行相同的标准化处理。
4. 模型测试与评估（Model Testing and Evaluation）
    (1) 测试集预测（Test Set Prediction）：
    使用最终训练好的模型，对保留的 5 条测试数据进行预测。
    计算预测值与实际值的 MAE，评估模型的泛化能力。
    (2) 误差范围评估（Error Range Evaluation）：
    对 Etching Rate (nm/min) 预测值，判断其与实际值的误差是否在 2.5% 以内。
    对 Angle (°) 预测值，判断其与实际值的误差是否在 ±1° 以内。
    (3) 可视化（Visualization）：
    绘制 Etching Rate 和 Angle 的实际值与预测值的散点图，展示模型的预测效果。
    通过图表展示模型在测试集上的表现，直观了解模型预测与实际值之间的关系。
5. 训练与测试结果总结（Training and Testing Results Summary）
    交叉验证平均 MAE：在 5 折交叉验证中，平均 MAE 为 X（需填入实际值），说明模型在训练数据上的整体表现。
    测试集 MAE：在测试集上的 MAE 为 Y（需填入实际值），代表模型在未见过的数据上的表现。
    预测误差范围：
    Etching Rate (nm/min)：Z 条样本（需填入实际值）在 2.5% 的误差范围内。
    Angle (°)：W 条样本（需填入实际值）在 ±1° 的误差范围内。
"""

# 设置环境变量以关闭 oneDNN 提示信息（可选）
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 检查 TensorFlow 版本
print(f"TensorFlow Version: {tf.__version__}")
print("Keras is available in TensorFlow:", hasattr(tf, 'keras'))

# 数据加载
data_file = 'test.csv'  # 假设数据文件名为 test.csv
if not os.path.exists(data_file):
    raise FileNotFoundError(f"The file '{data_file}' does not exist. Please check the path and file name.")

# 读取数据
data = pd.read_csv(data_file)

# 字段名转换为英文（如需）
data.columns = [
    'Etching Material', 'CD (um)', 'Pressure (mT)', 'SRF-C (W)', 'SRF-E (W)',
    'BRF (W)', 'SF6 (sccm)', 'CF4 (sccm)', 'CHF3 (sccm)', 'C4F8 (sccm)',
    'N2 (sccm)', 'O2 (sccm)', 'Ar (sccm)', 'He Pressure (T)', 'Chiller Temp (℃)',
    'Time (s)', 'Etching Depth (nm)', 'Etching Rate (nm/min)',
    'PR Selectivity', 'Angle (°)', 'Uniformity (%)'
]

# 删除不需要的特征
data = data.drop(columns=['CF4 (sccm)', 'CHF3 (sccm)', 'C4F8 (sccm)', 'N2 (sccm)', 'O2 (sccm)'])

# 选择输入特征和输出特征
input_features = [
    'Pressure (mT)', 'SRF-C (W)', 'SRF-E (W)', 'BRF (W)', 'SF6 (sccm)',
    'Ar (sccm)', 'He Pressure (T)', 'Chiller Temp (℃)', 'Time (s)'
]
output_features = ['Etching Rate (nm/min)', 'Angle (°)']

# 提取输入和输出数据
X = data[input_features]
y = data[output_features]

# 手动划分测试集（保留5条数据）
test_indices = [0, 1, 2, 3, 4]  # 选择前5条作为测试集，可以根据需要修改索引
X_test = X.iloc[test_indices]
y_test = y.iloc[test_indices]

# 剩余数据作为训练和验证集
X_train_val = X.drop(test_indices, axis=0)
y_train_val = y.drop(test_indices, axis=0)

# 数据标准化
scaler_X = StandardScaler().fit(X_train_val)
X_train_val_scaled = scaler_X.transform(X_train_val)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler().fit(y_train_val)
y_train_val_scaled = scaler_y.transform(y_train_val)
y_test_scaled = scaler_y.transform(y_test)

# 使用交叉验证（KFold）
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# 用于保存交叉验证的结果
cv_results = []
fold_no = 1

for train_index, val_index in kf.split(X_train_val_scaled):
    # 在27条数据上划分训练集和验证集（3/7分）
    X_train, X_val = X_train_val_scaled[train_index], X_train_val_scaled[val_index]
    y_train, y_val = y_train_val_scaled[train_index], y_train_val_scaled[val_index]

    # 构建神经网络模型
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(2)  # 输出为2个特征，分别为刻蚀速率和角度
    ])

    # 编译模型
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    # 训练模型
    history = model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=0)

    # 验证集评估
    y_val_pred_scaled = model.predict(X_val)
    val_mae = mean_absolute_error(y_val, y_val_pred_scaled)
    print(f'Fold {fold_no} Validation MAE: {val_mae:.4f}')

    # 保存每次验证结果
    fold_results = {
        'Fold': fold_no,
        'MAE': val_mae
    }
    cv_results.append(fold_results)

    fold_no += 1

# 计算交叉验证的平均 MAE
mean_val_mae = np.mean([result['MAE'] for result in cv_results])
print(f'5-Fold Cross Validation Mean MAE: {mean_val_mae:.4f}')

# 使用所有27条数据训练最终模型
model.fit(X_train_val_scaled, y_train_val_scaled, epochs=100, batch_size=16, verbose=0)
model.save('v1.h5')
print("模型已保存为 'v1.h5'")

# 保存标准化参数
np.save('models/scaler_X_mean.npy', scaler_X.mean_)
np.save('models/scaler_X_scale.npy', scaler_X.scale_)
np.save('models/scaler_y_mean.npy', scaler_y.mean_)
np.save('models/scaler_y_scale.npy', scaler_y.scale_)

# 使用最终模型对保留的测试集进行预测
y_test_pred_scaled = model.predict(X_test_scaled)
y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled)

# 计算测试集 MAE
test_mae = mean_absolute_error(y_test, y_test_pred)
print(f'Test Set MAE: {test_mae:.4f}')

# 打印测试集实际值和预测值
print('测试集实际值:')
print(pd.DataFrame(y_test, columns=output_features))
print('测试集预测值:')
print(pd.DataFrame(y_test_pred, columns=output_features))

# 计算预测结果是否在指定误差范围内
acceptable_rate_diff = np.abs((y_test['Etching Rate (nm/min)'] - y_test_pred[:, 0]) / y_test['Etching Rate (nm/min)']) <= 0.025
acceptable_angle_diff = np.abs(y_test['Angle (°)'] - y_test_pred[:, 1]) <= 1

# 统计符合误差要求的样本数
acceptable_rate_count = np.sum(acceptable_rate_diff)
acceptable_angle_count = np.sum(acceptable_angle_diff)

print(f'Etching Rate 在 2.5% 误差范围内的样本数: {acceptable_rate_count} / {len(y_test)}')
print(f'Angle 在 ±1° 误差范围内的样本数: {acceptable_angle_count} / {len(y_test)}')

# 可视化实际值和预测值
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_test['Etching Rate (nm/min)'], y_test_pred[:, 0], color='skyblue', label='Etching Rate (nm/min)')
plt.plot([min(y_test['Etching Rate (nm/min)']), max(y_test['Etching Rate (nm/min)'])],
         [min(y_test['Etching Rate (nm/min)']), max(y_test['Etching Rate (nm/min)'])], color='blue', linestyle='--')
plt.title('Etching Rate: Actual vs Predicted')
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(y_test['Angle (°)'], y_test_pred[:, 1], color='orange', label='Angle (°)')
plt.plot([min(y_test['Angle (°)']), max(y_test['Angle (°)'])],
         [min(y_test['Angle (°)']), max(y_test['Angle (°)'])], color='red', linestyle='--')
plt.title('Angle: Actual vs Predicted')
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')
plt.legend()

plt.tight_layout()
plt.show()

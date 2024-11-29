from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# 文件路径配置
DATA_DIR = 'data'
MODEL_DIR = 'models'
MODEL_FILE = os.path.join(MODEL_DIR, 'v3.h5')
uploaded_data = None
model = None
scaler_X = None
scaler_y = None
# 定义特征顺序，与前端一致
feature_order = [
    'Pressure (mT)', 'SRF-C (W)', 'SRF-E (W)', 'BRF (W)', 'SF6 (sccm)',
    'Ar (sccm)', 'He Pressure (T)', 'Chiller Temp (℃)', 'Time (s)'
]
# 首页
@app.route('/')
def index():
    return render_template('index.html')

# 上传数据
@app.route('/upload_data', methods=['POST'])
def upload_data():
    global uploaded_data
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    file_path = os.path.join(DATA_DIR, 'uploaded.csv')
    file.save(file_path)

    try:
        data = pd.read_csv(file_path)
    except Exception as e:
        return jsonify({'error': f'Failed to read file: {str(e)}'}), 400

    # 删除无效列和全为空的行
    data = data.drop(columns=['Uniformity (%)'], errors='ignore').dropna(how='all')
    uploaded_data = data

    # 默认切分数据集：前 5 条为测试集，其余为训练集
    test_set = data.iloc[:5].to_dict(orient='records')
    train_set = data.iloc[5:].to_dict(orient='records')

    return jsonify({
        'message': 'File uploaded successfully',
        'total_rows': len(data),
        'columns': list(data.columns),
        'default_test_set': test_set,
        'default_train_set': train_set,
    })


# 动态划分数据集
@app.route('/split_data', methods=['POST'])
def split_data():
    global uploaded_data
    if uploaded_data is None:
        return jsonify({'error': 'No data uploaded'}), 400

    test_size = int(request.json.get('test_size', 5))
    if test_size < 1 or test_size >= len(uploaded_data):
        return jsonify({'error': 'Invalid test size'}), 400

    # 切分数据集
    test_set = uploaded_data.iloc[:test_size].to_dict(orient='records')
    train_set = uploaded_data.iloc[test_size:].to_dict(orient='records')

    return jsonify({
        'test_set': test_set,
        'train_set': train_set,
    })


# 训练模型
# @app.route('/train_model', methods=['POST'])
# def train_model():
#     global model, scaler_X, scaler_y
#
#     # 接收训练集和测试集信息
#     train_set = pd.DataFrame(request.json.get('train_set'))
#     test_set = pd.DataFrame(request.json.get('test_set'))
#     epochs = request.json.get('epochs', 10)
#
#     if train_set.empty or test_set.empty:
#         return jsonify({'error': 'Training or test set is missing'}), 400
#
#     # 提取输入特征和目标
#     input_features = [  'Pressure (mT)', 'SRF-C (W)', 'SRF-E (W)', 'BRF (W)', 'SF6 (sccm)', 'Ar (sccm)',
#                       'He Pressure (T)', 'Chiller Temp (℃)', 'Time (s)']
#     output_features = ['Etching Rate (nm/min)', 'Angle (°)']
#
#     X_train = train_set[input_features]
#     y_train = train_set[output_features]
#
#     # 数据标准化
#     scaler_X = StandardScaler().fit(X_train)
#     scaler_y = StandardScaler().fit(y_train)
#     X_train_scaled = scaler_X.transform(X_train)
#     y_train_scaled = scaler_y.transform(y_train)
#
#     # 构建并训练模型
#     model = tf.keras.Sequential([
#         tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
#         tf.keras.layers.Dropout(0.3),
#         tf.keras.layers.Dense(64, activation='relu'),
#         tf.keras.layers.Dense(len(output_features))
#     ])
#     model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
#
#     # 训练过程日志
#     training_log = []
#     for epoch in range(epochs):
#         history = model.fit(X_train_scaled, y_train_scaled, epochs=1, batch_size=16, verbose=0)
#         loss = history.history['loss'][0]
#         mae = history.history['mae'][0]
#         training_log.append(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f}, MAE: {mae:.4f}")
#
#     # 保存模型和标准化参数
#     model.save(MODEL_FILE)
#     np.save(os.path.join(MODEL_DIR, 'scaler_X_mean.npy'), scaler_X.mean_)
#     np.save(os.path.join(MODEL_DIR, 'scaler_X_scale.npy'), scaler_X.scale_)
#     np.save(os.path.join(MODEL_DIR, 'scaler_y_mean.npy'), scaler_y.mean_)
#     np.save(os.path.join(MODEL_DIR, 'scaler_y_scale.npy'), scaler_y.scale_)
#
#     return jsonify({
#         'message': 'Model trained successfully',
#         'train_size': len(X_train),
#         'test_size': len(test_set),
#         'log': training_log
#     })

# 训练模型
@app.route('/train_model', methods=['POST'])
def train_model():
    global model, scaler_X, scaler_y

    # 接收训练集和测试集信息
    train_set = pd.DataFrame(request.json.get('train_set'))
    test_set = pd.DataFrame(request.json.get('test_set'))
    epochs = request.json.get('epochs', 10)

    if train_set.empty or test_set.empty:
        return jsonify({'error': 'Training or test set is missing'}), 400

    # 提取输入特征和目标
    input_features = [  'Pressure (mT)', 'SRF-C (W)', 'SRF-E (W)', 'BRF (W)', 'SF6 (sccm)', 'Ar (sccm)',
                      'He Pressure (T)', 'Chiller Temp (℃)', 'Time (s)']
    output_features = ['Etching Rate (nm/min)', 'Angle (°)']

    X_train = train_set[input_features]
    y_train = train_set[output_features]

    # 数据标准化
    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train)
    X_train_scaled = scaler_X.transform(X_train)
    y_train_scaled = scaler_y.transform(y_train)

    # 构建并训练模型
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(len(output_features))
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    # 保存每个 epoch 的损失
    losses = []
    maes = []

    # 手动训练并记录损失
    for epoch in range(epochs):
        history = model.fit(X_train_scaled, y_train_scaled, epochs=1, batch_size=16, verbose=0)
        loss = history.history['loss'][0]
        mae = history.history['mae'][0]
        losses.append(loss)
        maes.append(mae)

    # 保存模型和标准化参数
    model.save(MODEL_FILE)
    np.save(os.path.join(MODEL_DIR, 'scaler_X_mean.npy'), scaler_X.mean_)
    np.save(os.path.join(MODEL_DIR, 'scaler_X_scale.npy'), scaler_X.scale_)
    np.save(os.path.join(MODEL_DIR, 'scaler_y_mean.npy'), scaler_y.mean_)
    np.save(os.path.join(MODEL_DIR, 'scaler_y_scale.npy'), scaler_y.scale_)

    return jsonify({
        'message': 'Model trained successfully',
        'train_size': len(X_train),
        'test_size': len(test_set),
        'losses': losses,
        'maes': maes
    })



# 预测数据
@app.route('/predict', methods=['POST'])
def predict():
    global model, scaler_X, scaler_y, feature_order  # 引用全局变量
    if model is None or scaler_X is None or scaler_y is None:
        return jsonify({'error': 'Model is not trained'}), 400

    # 接收输入数据
    input_data = request.json.get('data')
    if not input_data or len(input_data) != len(feature_order):
        return jsonify({'error': f'Invalid input data. Expected {len(feature_order)} features'}), 400

    # 标准化输入数据
    X_input = np.array(input_data).reshape(1, -1)
    X_scaled = scaler_X.transform(X_input)

    # 模型预测并反标准化
    y_scaled_pred = model.predict(X_scaled)
    y_pred = scaler_y.inverse_transform(y_scaled_pred)

    return jsonify({'prediction': y_pred.tolist()})



if __name__ == '__main__':
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    app.run(debug=True)

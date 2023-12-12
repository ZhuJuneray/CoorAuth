import numpy as np
import pandas as pd
from keras import regularizers
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from tensorflow.keras import layers, models
import os
os.chdir(os.path.join(os.getcwd(),'VRAuth 2'))

features_sli = 300

def smooth_data(arr, window_parameter=31, polyorder_parameter=2):
    arr_smoothed = savgol_filter(arr, window_length=window_parameter, polyorder=polyorder_parameter)
    return arr_smoothed

def difference_yaw_gaze_head(user,date,num):
    data1 = pd.read_csv(os.path.join("unity_processed_data","GazeCalculate_data_" + user + "-" + date + "-" + str(num+1) + "_unity_processed.csv"))
    data2 = pd.read_csv(os.path.join("unity_processed_data","Head_data_" + user + "-" + date + "-" + str(num+1) + "_unity_processed.csv"))
    # Create DataFrame
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)

    L_Yaw_zero = df1['L_Yaw'][0]
    L_Yaw_df = [x - L_Yaw_zero if abs(x - L_Yaw_zero) < 200 else (x - L_Yaw_zero -  360 if x- L_Yaw_zero >200 else x - L_Yaw_zero + 360 ) for x in df1['L_Yaw']]
    Yaw_zero = df2['Yaw'][0] if df2['Yaw'][0] < 180 else df2['Yaw'][0] - 360
    Yaw_df = [x - Yaw_zero if x < 180 else x - Yaw_zero -  360 for x in df2['Yaw']]

    return [x - y for x, y in zip(L_Yaw_df, Yaw_df)]

def extract_features(sequence): # 把序列切成十段，每段取均值、最大值、最小值、方差，共40个特征，返回一个拼接的一维数组
    # 计算每个子序列的基本长度和额外长度
    n = len(sequence)
    sub_seq_length = n // 10
    remainder = n % 10

    # 初始化特征数组
    features = []
    features_mean = []
    features_max = []
    features_min = []
    features_var = []
    start = 0

    # 对每个子序列进行迭代
    for i in range(10):
        # 调整子序列长度
        end = start + sub_seq_length + (1 if i < remainder else 0)
        sub_seq = sequence[start:end]

        # 计算特征
        mean = np.mean(sub_seq)
        max_value = np.max(sub_seq)
        min_value = np.min(sub_seq)
        variance = np.var(sub_seq)

        # 添加到特征数组
        
        features_mean.append(mean)
        features_max.append(max_value)
        features_min.append(min_value)
        features_var.append(variance)
        # print("features: ", features)

        # 更新起始位置
        start = end

    return np.concatenate([features_mean,features_max, features_min, features_var])


# 示例数据生成
authentications_per_person = 6
features_sli = 300
user_names = ['zjr', 'zs', 'gj', 'pyj']
num_people = len(user_names)
dates = ['1118']

# # 特征
# result_array = np.array([])
# for user in user_names:
#     for num in range(authentications_per_person):
#         for date in dates:
#             execute_block = True # skip specific line drawing: True for skipping and False for drawing all line
#             if execute_block:
#                 if (date=="1108" or date=="1109") and (user=='jyc' or user=='lhy' or user=='wgh' or user=='yf'):
#                 # if user=='zjr':
#                     continue
#             # Head
#             data_head=pd.read_csv("E:\Desktop\data\VRAuth\Head_data_"+user+'-'+date+'-'+str(num+1)+'.csv')
#             QuaternionX_data = data_head['H-QuaternionX']
#             QuaternionX_data_smoothed = smooth_data(QuaternionX_data)
#             d1 = np.array(QuaternionX_data_smoothed[:features_sli])
#             QuaternionY_data = data_head['H-QuaternionY']
#             QuaternionY_data_smoothed = smooth_data(QuaternionY_data)
#             d2 = np.array(QuaternionY_data_smoothed[:features_sli])
#             QuaternionZ_data = data_head['H-QuaternionZ']
#             QuaternionZ_data_smoothed = smooth_data(QuaternionZ_data)
#             d3 = np.array(QuaternionZ_data_smoothed[:features_sli])
#             QuaternionW_data = data_head['H-QuaternionW']
#             QuaternionW_data_smoothed = smooth_data(QuaternionW_data)
#             d4 = np.array(QuaternionW_data_smoothed[:features_sli])

#             Vector3X_data = data_head['H-Vector3X']
#             Vector3X_data_smoothed = smooth_data(Vector3X_data)
#             v1 = np.array(Vector3X_data_smoothed[:features_sli])
#             Vector3Y_data = data_head['H-Vector3Y']
#             Vector3Y_data_smoothed = smooth_data(Vector3Y_data)
#             v2 = np.array(Vector3Y_data_smoothed[:features_sli])
#             Vector3Z_data = data_head['H-Vector3Z']
#             Vector3Z_data_smoothed = smooth_data(Vector3Z_data)
#             v3 = np.array(Vector3Z_data_smoothed[:features_sli])

#             # Eye points
#             data_eye = pd.read_csv("E:\Desktop\data\VRAuth\GazeRaw_data_"+user+'-'+date+'-'+str(num+1)+'.csv')
#             QuaternionX_data = data_eye['L-QuaternionX']
#             QuaternionX_data_smoothed = smooth_data(QuaternionX_data)
#             d1_el = np.array(QuaternionX_data_smoothed[:features_sli])
#             QuaternionY_data = data_eye['L-QuaternionY']
#             QuaternionY_data_smoothed = smooth_data(QuaternionY_data)
#             d2_el = np.array(QuaternionY_data_smoothed[:features_sli])
#             QuaternionZ_data = data_eye['L-QuaternionZ']
#             QuaternionZ_data_smoothed = smooth_data(QuaternionZ_data)
#             d3_el = np.array(QuaternionZ_data_smoothed[:features_sli])
#             QuaternionW_data = data_eye['L-QuaternionW']
#             QuaternionW_data_smoothed = smooth_data(QuaternionW_data)
#             d4_el = np.array(QuaternionW_data_smoothed[:features_sli])

#             QuaternionX_data = data_eye['R-QuaternionX']
#             QuaternionX_data_smoothed = smooth_data(QuaternionX_data)
#             d1_er = np.array(QuaternionX_data_smoothed[:features_sli])
#             QuaternionY_data = data_eye['R-QuaternionY']
#             QuaternionY_data_smoothed = smooth_data(QuaternionY_data)
#             d2_er = np.array(QuaternionY_data_smoothed[:features_sli])
#             QuaternionZ_data = data_eye['R-QuaternionZ']
#             QuaternionZ_data_smoothed = smooth_data(QuaternionZ_data)
#             d3_er = np.array(QuaternionZ_data_smoothed[:features_sli])
#             QuaternionW_data = data_eye['R-QuaternionW']
#             QuaternionW_data_smoothed = smooth_data(QuaternionW_data)
#             d4_er = np.array(QuaternionW_data_smoothed[:features_sli])

#             merged_array = np.concatenate([d1, d2, d3, d4, v1, v2, v3, d1_el, d2_el, d3_el, d4_el, d1_er, d2_er, d3_er, d4_er])
#             # merged_array = np.concatenate([d1, d2, d3, d4])
#             # merged_array = np.concatenate([d1_el, d2_el, d3_el, d4_el, d1_er, d2_er, d3_er, d4_er])
#             result_array = np.vstack([result_array, merged_array]) if result_array.size else merged_array

# 特征
result_array = np.array([])
for user in user_names:
    for num in range(authentications_per_person):
        for date in dates:
            execute_block = True # skip specific line drawing: True for skipping and False for drawing all line
            if execute_block:
                if (date=="1118") and (user=='zhao') and (num==5):
                # if user=='zjr':
                    continue
            # Head
            data_head=pd.read_csv("Head_data_"+user+'-'+date+'-'+str(num+1)+'.csv')
            QuaternionX_data = data_head['H-QuaternionX']
            QuaternionX_data_smoothed = smooth_data(QuaternionX_data)
            d1 = np.array(QuaternionX_data_smoothed[:features_sli])
            d1_feat = extract_features(np.array(QuaternionX_data))
            QuaternionY_data = data_head['H-QuaternionY']
            QuaternionY_data_smoothed = smooth_data(QuaternionY_data)
            d2 = np.array(QuaternionY_data_smoothed[:features_sli])
            d2_feat = extract_features(np.array(QuaternionY_data))
            QuaternionZ_data = data_head['H-QuaternionZ']
            QuaternionZ_data_smoothed = smooth_data(QuaternionZ_data)
            d3 = np.array(QuaternionZ_data_smoothed[:features_sli])
            d3_feat = extract_features(np.array(QuaternionZ_data))
            QuaternionW_data = data_head['H-QuaternionW']
            QuaternionW_data_smoothed = smooth_data(QuaternionW_data)
            d4 = np.array(QuaternionW_data_smoothed[:features_sli])
            d4_feat = extract_features(np.array(QuaternionW_data))

            Vector3X_data = data_head['H-Vector3X']
            Vector3X_data_smoothed = smooth_data(Vector3X_data)
            v1 = np.array(Vector3X_data_smoothed[:features_sli])
            v1_feat = extract_features(np.array(Vector3X_data))
            Vector3Y_data = data_head['H-Vector3Y']
            Vector3Y_data_smoothed = smooth_data(Vector3Y_data)
            v2 = np.array(Vector3Y_data_smoothed[:features_sli])
            v2_feat = extract_features(np.array(Vector3Y_data))
            Vector3Z_data = data_head['H-Vector3Z']
            Vector3Z_data_smoothed = smooth_data(Vector3Z_data)
            v3 = np.array(Vector3Z_data_smoothed[:features_sli])
            v3_feat = extract_features(np.array(Vector3Z_data))

            # Eye points
            data_eye = pd.read_csv("GazeRaw_data_"+user+'-'+date+'-'+str(num+1)+'.csv')
            QuaternionX_data = data_eye['L-QuaternionX']
            QuaternionX_data_smoothed = smooth_data(QuaternionX_data)
            d1_el = np.array(QuaternionX_data_smoothed[:features_sli])
            d1_el_feat = extract_features(np.array(QuaternionX_data))
            QuaternionY_data = data_eye['L-QuaternionY']
            QuaternionY_data_smoothed = smooth_data(QuaternionY_data)
            d2_el = np.array(QuaternionY_data_smoothed[:features_sli])
            d2_el_feat = extract_features(np.array(QuaternionY_data))
            QuaternionZ_data = data_eye['L-QuaternionZ']
            QuaternionZ_data_smoothed = smooth_data(QuaternionZ_data)
            d3_el = np.array(QuaternionZ_data_smoothed[:features_sli])
            d3_el_feat = extract_features(np.array(QuaternionZ_data))
            QuaternionW_data = data_eye['L-QuaternionW']
            QuaternionW_data_smoothed = smooth_data(QuaternionW_data)
            d4_el = np.array(QuaternionW_data_smoothed[:features_sli])
            d4_el_feat = extract_features(np.array(QuaternionW_data))

            QuaternionX_data = data_eye['R-QuaternionX']
            QuaternionX_data_smoothed = smooth_data(QuaternionX_data)
            d1_er = np.array(QuaternionX_data_smoothed[:features_sli])
            d1_er_feat = extract_features(np.array(QuaternionX_data))
            QuaternionY_data = data_eye['R-QuaternionY']
            QuaternionY_data_smoothed = smooth_data(QuaternionY_data)
            d2_er = np.array(QuaternionY_data_smoothed[:features_sli])
            d2_er_feat = extract_features(np.array(QuaternionY_data))
            QuaternionZ_data = data_eye['R-QuaternionZ']
            QuaternionZ_data_smoothed = smooth_data(QuaternionZ_data)
            d3_er = np.array(QuaternionZ_data_smoothed[:features_sli])
            d3_er_feat = extract_features(np.array(QuaternionZ_data))
            QuaternionW_data = data_eye['R-QuaternionW']
            QuaternionW_data_smoothed = smooth_data(QuaternionW_data)
            d4_er = np.array(QuaternionW_data_smoothed[:features_sli])
            d4_er_feat = extract_features(QuaternionW_data[:features_sli])

            
            # Head and eye points
            difference_yaw_gaze_head_data = difference_yaw_gaze_head(user,date,num)

            # merged_array = np.concatenate([d1, d1_feat, d2, d2_feat, d3, d3_feat, d4, d4_feat, v1, v1_feat, v2, v2_feat, v3, v3_feat, d1_el, d1_el_feat, d2_el, d2_el_feat, d3_el, d3_el_feat, d4_el, d4_el_feat, d1_er, d1_er_feat, d2_er, d2_er_feat,  d3_er, d3_er_feat,  d4_er, d4_er_feat, difference_yaw_gaze_head_data[:features_sli]])
            merged_array = np.concatenate([d1_feat, d2_feat, d3_feat, d4_feat, v1_feat, v2_feat, v3_feat, d1_el_feat, d2_el_feat, d3_el_feat, d4_el_feat, d1_er_feat, d2_er_feat, d3_er_feat, d4_er_feat, ])
            # print(d1)
            # merged_array = np.concatenate(
            #     [v1])
            result_array = np.vstack([result_array, merged_array]) if result_array.size else merged_array


# 生成示例数据
labels = np.repeat(np.arange(num_people), authentications_per_person)
data = result_array

# 数据标准化
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
print(data_scaled.shape[1])
# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(data_scaled, labels, test_size=0.2)

# 转换标签为独热编码
y_train_one_hot = tf.keras.utils.to_categorical(y_train, num_classes=num_people)
y_test_one_hot = tf.keras.utils.to_categorical(y_test, num_classes=num_people)

# # 构建神经网络模型
# model = models.Sequential()
# model.add(layers.Dense(32, activation='relu', input_shape=(data_scaled.shape[1],)))
# model.add(layers.Dense(num_people, activation='softmax'))

# 构建神经网络模型
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(data_scaled.shape[1],)))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(num_people, activation='softmax'))

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train_one_hot, epochs=30, batch_size=50, validation_split=0.1)

# 在测试集上评估模型
accuracy = model.evaluate(X_test, y_test_one_hot)[1]
print("Test accuracy:", accuracy)

# 对测试数据进行预测
predictions = model.predict(X_test)

## 打印每个测试样本的 softmax 概率结果和真实、预测的 labels
for i, (probs, true_label, predicted_label) in enumerate(zip(predictions, y_test, np.argmax(predictions, axis=1))):
    formatted_probs = [f"{prob:.4f}" for prob in probs]
    print(f"Sample {i+1} - Softmax Probabilities: {formatted_probs}, True Label: {true_label}, Predicted Label: {predicted_label}")

################################################################
# 攻击
attacker = 'zs'
data_head=pd.read_csv('E:\Desktop\data\VRAuth\Head_data_' + attacker + '-1118-gz-1.csv')
# data_head=pd.read_csv("E:\Desktop\data\VRAuth\Head_data_gj-1118-1.csv")
QuaternionX_data = data_head['H-QuaternionX']
QuaternionX_data_smoothed = smooth_data(QuaternionX_data)
d1 = np.array(QuaternionX_data_smoothed[:features_sli])
QuaternionY_data = data_head['H-QuaternionY']
QuaternionY_data_smoothed = smooth_data(QuaternionY_data)
d2 = np.array(QuaternionY_data_smoothed[:features_sli])
QuaternionZ_data = data_head['H-QuaternionZ']
QuaternionZ_data_smoothed = smooth_data(QuaternionZ_data)
d3 = np.array(QuaternionZ_data_smoothed[:features_sli])
QuaternionW_data = data_head['H-QuaternionW']
QuaternionW_data_smoothed = smooth_data(QuaternionW_data)
d4 = np.array(QuaternionW_data_smoothed[:features_sli])

Vector3X_data = data_head['H-Vector3X']
Vector3X_data_smoothed = smooth_data(Vector3X_data)
v1 = np.array(Vector3X_data_smoothed[:features_sli])
Vector3Y_data = data_head['H-Vector3Y']
Vector3Y_data_smoothed = smooth_data(Vector3Y_data)
v2 = np.array(Vector3Y_data_smoothed[:features_sli])
Vector3Z_data = data_head['H-Vector3Z']
Vector3Z_data_smoothed = smooth_data(Vector3Z_data)
v3 = np.array(Vector3Z_data_smoothed[:features_sli])

# Eye points
data_eye = pd.read_csv('E:\Desktop\data\VRAuth\GazeRaw_data_' + attacker + '-1118-gz-1.csv')
# data_eye=pd.read_csv("E:\Desktop\data\VRAuth\GazeRaw_data_gj-1118-1.csv")
QuaternionX_data = data_eye['L-QuaternionX']
QuaternionX_data_smoothed = smooth_data(QuaternionX_data)
d1_el = np.array(QuaternionX_data_smoothed[:features_sli])
QuaternionY_data = data_eye['L-QuaternionY']
QuaternionY_data_smoothed = smooth_data(QuaternionY_data)
d2_el = np.array(QuaternionY_data_smoothed[:features_sli])
QuaternionZ_data = data_eye['L-QuaternionZ']
QuaternionZ_data_smoothed = smooth_data(QuaternionZ_data)
d3_el = np.array(QuaternionZ_data_smoothed[:features_sli])
QuaternionW_data = data_eye['L-QuaternionW']
QuaternionW_data_smoothed = smooth_data(QuaternionW_data)
d4_el = np.array(QuaternionW_data_smoothed[:features_sli])

QuaternionX_data = data_eye['R-QuaternionX']
QuaternionX_data_smoothed = smooth_data(QuaternionX_data)
d1_er = np.array(QuaternionX_data_smoothed[:features_sli])
QuaternionY_data = data_eye['R-QuaternionY']
QuaternionY_data_smoothed = smooth_data(QuaternionY_data)
d2_er = np.array(QuaternionY_data_smoothed[:features_sli])
QuaternionZ_data = data_eye['R-QuaternionZ']
QuaternionZ_data_smoothed = smooth_data(QuaternionZ_data)
d3_er = np.array(QuaternionZ_data_smoothed[:features_sli])
QuaternionW_data = data_eye['R-QuaternionW']
QuaternionW_data_smoothed = smooth_data(QuaternionW_data)
d4_er = np.array(QuaternionW_data_smoothed[:features_sli])

merged_array = np.concatenate([d1, d2, d3, d4, v1, v2, v3, d1_el, d2_el, d3_el, d4_el, d1_er, d2_er, d3_er, d4_er])
# merged_array = np.concatenate([d1, d2, d3, d4, v1, v2, v3])
# merged_array = np.concatenate([d1_el, d2_el, d3_el, d4_el, d1_er, d2_er, d3_er, d4_er])
# 数据标准化
attacker_data_scaled = scaler.transform(merged_array.reshape(1, -1))

# 获取攻击者数据的标签
attacker_labels = np.array([0])

# 转换标签为独热编码
attacker_labels_one_hot = tf.keras.utils.to_categorical(attacker_labels, num_classes=num_people)

# 在模型上评估攻击者的数据
attacker_loss = model.evaluate(attacker_data_scaled, attacker_labels_one_hot)[0]
print("Attacker's Data Loss:", attacker_loss)


################################################################
# 测试者
#data_head=pd.read_csv("E:\Desktop\data\VRAuth\Head_data_zs-1118-gz-1.csv")
tester = 'zs'
data_head=pd.read_csv("E:\Desktop\data\VRAuth\Head_data_" + tester + "-1118-2.csv")
QuaternionX_data = data_head['H-QuaternionX']
QuaternionX_data_smoothed = smooth_data(QuaternionX_data)
d1 = np.array(QuaternionX_data_smoothed[:features_sli])
QuaternionY_data = data_head['H-QuaternionY']
QuaternionY_data_smoothed = smooth_data(QuaternionY_data)
d2 = np.array(QuaternionY_data_smoothed[:features_sli])
QuaternionZ_data = data_head['H-QuaternionZ']
QuaternionZ_data_smoothed = smooth_data(QuaternionZ_data)
d3 = np.array(QuaternionZ_data_smoothed[:features_sli])
QuaternionW_data = data_head['H-QuaternionW']
QuaternionW_data_smoothed = smooth_data(QuaternionW_data)
d4 = np.array(QuaternionW_data_smoothed[:features_sli])

Vector3X_data = data_head['H-Vector3X']
Vector3X_data_smoothed = smooth_data(Vector3X_data)
v1 = np.array(Vector3X_data_smoothed[:features_sli])
Vector3Y_data = data_head['H-Vector3Y']
Vector3Y_data_smoothed = smooth_data(Vector3Y_data)
v2 = np.array(Vector3Y_data_smoothed[:features_sli])
Vector3Z_data = data_head['H-Vector3Z']
Vector3Z_data_smoothed = smooth_data(Vector3Z_data)
v3 = np.array(Vector3Z_data_smoothed[:features_sli])

# Eye points
# data_eye = pd.read_csv("E:\Desktop\data\VRAuth\GazeRaw_data_zs-1118-gz-1.csv")
data_eye=pd.read_csv("E:\Desktop\data\VRAuth\GazeRaw_data_" + tester +"-1118-2.csv")
QuaternionX_data = data_eye['L-QuaternionX']
QuaternionX_data_smoothed = smooth_data(QuaternionX_data)
d1_el = np.array(QuaternionX_data_smoothed[:features_sli])
QuaternionY_data = data_eye['L-QuaternionY']
QuaternionY_data_smoothed = smooth_data(QuaternionY_data)
d2_el = np.array(QuaternionY_data_smoothed[:features_sli])
QuaternionZ_data = data_eye['L-QuaternionZ']
QuaternionZ_data_smoothed = smooth_data(QuaternionZ_data)
d3_el = np.array(QuaternionZ_data_smoothed[:features_sli])
QuaternionW_data = data_eye['L-QuaternionW']
QuaternionW_data_smoothed = smooth_data(QuaternionW_data)
d4_el = np.array(QuaternionW_data_smoothed[:features_sli])

QuaternionX_data = data_eye['R-QuaternionX']
QuaternionX_data_smoothed = smooth_data(QuaternionX_data)
d1_er = np.array(QuaternionX_data_smoothed[:features_sli])
QuaternionY_data = data_eye['R-QuaternionY']
QuaternionY_data_smoothed = smooth_data(QuaternionY_data)
d2_er = np.array(QuaternionY_data_smoothed[:features_sli])
QuaternionZ_data = data_eye['R-QuaternionZ']
QuaternionZ_data_smoothed = smooth_data(QuaternionZ_data)
d3_er = np.array(QuaternionZ_data_smoothed[:features_sli])
QuaternionW_data = data_eye['R-QuaternionW']
QuaternionW_data_smoothed = smooth_data(QuaternionW_data)
d4_er = np.array(QuaternionW_data_smoothed[:features_sli])

merged_array = np.concatenate([d1, d2, d3, d4, v1, v2, v3, d1_el, d2_el, d3_el, d4_el, d1_er, d2_er, d3_er, d4_er])
# merged_array = np.concatenate([d1, d2, d3, d4, v1, v2, v3])
# merged_array = np.concatenate([d1_el, d2_el, d3_el, d4_el, d1_er, d2_er, d3_er, d4_er])
# 数据标准化
attacker_data_scaled = scaler.transform(merged_array.reshape(1, -1))

attacker_labels = np.array([1])

# 转换标签为独热编码
attacker_labels_one_hot = tf.keras.utils.to_categorical(attacker_labels, num_classes=num_people)

# 在模型上评估攻击者的数据
loss = model.evaluate(attacker_data_scaled, attacker_labels_one_hot)
print("User's Data Loss:", loss[0])
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, classification_report
from sklearn.model_selection import StratifiedKFold

def smooth_data(arr, window_parameter=31, polyorder_parameter=2):
    arr_smoothed = savgol_filter(arr, window_length=window_parameter, polyorder=polyorder_parameter)
    return arr_smoothed

# 示例数据生成
authentications_per_person = 6
features_sli = 300
user_names=['zs', 'zjr', 'gj', 'pyj']
num_people = len(user_names)
dates=['1118']

# 特征
result_array = np.array([])
for user in user_names:
    for num in range(authentications_per_person):
        for date in dates:
            execute_block = True # skip specific line drawing: True for skipping and False for drawing all line
            if execute_block:
                if (date=="1108" or date=="1109") and (user=='jyc' or user=='lhy' or user=='wgh' or user=='yf'):
                # if user=='zjr':
                    continue
            # Head
            data_head=pd.read_csv("E:\Desktop\data\VRAuth\Head_data_"+user+'-'+date+'-'+str(num+1)+'.csv')
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
            data_eye = pd.read_csv("E:\Desktop\data\VRAuth\GazeRaw_data_"+user+'-'+date+'-'+str(num+1)+'.csv')
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
            # merged_array = np.concatenate(
            #     [v1])
            result_array = np.vstack([result_array, merged_array]) if result_array.size else merged_array

# 生成示例数据
labels = np.repeat(np.arange(num_people), authentications_per_person)
data = result_array
# 打印示例数据形状
print("Data shape:", data.shape)
print("Labels shape:", labels.shape)

# 数据标准化
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(data_scaled, labels, test_size=0.2)
print("testing shape:", X_test.shape)
# 创建SVM模型
svm_model = SVC(kernel='linear', C=2)
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)

# 准确度
accuracy = accuracy_score(y_test, y_pred)
print("准确度:", accuracy)

# 混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
print("混淆矩阵:")
print(conf_matrix)

# 精确度、召回率、F1分数
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
print("精确度:", precision)
print("召回率:", recall)
print("F1分数:", f1)

# 分类报告
class_report = classification_report(y_test, y_pred)
print("分类报告:")
print(class_report)

########################################################################
# 交叉验证
for label in np.unique(labels):
    count = np.sum(labels == label)
    print(f"Class {label}: {count} samples")
# 设置交叉验证的折叠数
n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True)

# 初始化准确性列表，用于存储每个折叠的模型准确性
accuracies = []

for train_index, test_index in kf.split(data_scaled, labels):
    X_train, X_test = data_scaled[train_index], data_scaled[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    # 创建SVM模型
    svm_model = SVC(kernel='linear', C=1.0)

    # 模型训练
    svm_model.fit(X_train, y_train)

    # 模型预测
    y_pred = svm_model.predict(X_test)

    # 计算准确性
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

# 打印每个折叠的准确性
for i, acc in enumerate(accuracies):
    print(f"Fold {i+1} Accuracy: {acc}")

# 打印平均准确性
average_accuracy = np.mean(accuracies)
print("Average Accuracy:", average_accuracy)


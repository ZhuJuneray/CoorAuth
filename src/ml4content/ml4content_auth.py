import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, classification_report
import os

from sklearn.linear_model import LogisticRegression  # 导入逻辑回归模型，因为逻辑回归是二分类问题的常用模型。
from sklearn.svm import SVC


# os.chdir(os.path.join(os.getcwd(), 'data'))


def smooth_data(arr, window_parameter=31, polyorder_parameter=2):
    arr_smoothed = savgol_filter(arr, window_length=window_parameter, polyorder=polyorder_parameter)
    return arr_smoothed


def difference_yaw_gaze_head(user, date, num):
    data1 = pd.read_csv(os.path.join("unity_processed_data", "GazeCalculate_data_" + user + "-" + date + "-" + str(
        num + 1) + "_unity_processed.csv"))
    data2 = pd.read_csv(os.path.join("unity_processed_data",
                                     "Head_data_" + user + "-" + date + "-" + str(num + 1) + "_unity_processed.csv"))
    # Create DataFrame
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)

    L_Yaw_zero = df1['L_Yaw'][0]
    L_Yaw_df = [x - L_Yaw_zero if abs(x - L_Yaw_zero) < 200 else (
        x - L_Yaw_zero - 360 if x - L_Yaw_zero > 200 else x - L_Yaw_zero + 360) for x in df1['L_Yaw']]
    Yaw_zero = df2['Yaw'][0] if df2['Yaw'][0] < 180 else df2['Yaw'][0] - 360
    Yaw_df = [x - Yaw_zero if x < 180 else x - Yaw_zero - 360 for x in df2['Yaw']]

    return [x - y for x, y in zip(L_Yaw_df, Yaw_df)]


def extract_features(sequence):  # 把序列切成十段，每段取均值、最大值、最小值、方差，共40个特征，返回一个拼接的一维数组
    # 计算每个子序列的基本长度和额外长度
    n = len(sequence)
    # print("length" + str(n))
    slice_num = 10
    sub_seq_length = n // slice_num if n % slice_num == 0 else n // slice_num + 1# 向上取整
    remainder = sub_seq_length - (n // slice_num + 1) * slice_num + n # 处理最后一段未填充满

    # 初始化特征数组
    features = []
    features_mean = []
    features_max = []
    features_min = []
    features_var = []
    start = 0

    # 对每个子序列进行迭代
    for i in range(slice_num):
        # 调整子序列长度
        end = start
        if i < slice_num - 1:
            end = start + sub_seq_length
        else:
            end = start + remainder if remainder > 0 else sub_seq_length + start
        # print("start" + str(start) + " end" + str(end))
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

    return np.concatenate([features_mean, features_max, features_min, features_var])

def data_processing(authentications_per_person, user_names, dates, rotdir = None):
    # 特征
    result_array = np.array([])
    for user in user_names:
        for num in range(authentications_per_person):
            for date in dates:
                execute_block = True  # skip specific line drawing: True for skipping and False for drawing all line
                if execute_block:
                    if (date == "1118") and (user == 'zhao') and (num == 5):
                        # if user=='zjr':
                        continue
                # Head
                data_head = pd.read_csv(
                    rotdir + "\Head_data_" + user + '-' + date + '-' + str(num + 1) + '.csv')
                QuaternionX_data = data_head['H-QuaternionX']
                QuaternionX_data_smoothed = smooth_data(QuaternionX_data)
                d1 = np.array(QuaternionX_data_smoothed)
                d1_feat = extract_features(d1)
                QuaternionY_data = data_head['H-QuaternionY']
                QuaternionY_data_smoothed = smooth_data(QuaternionY_data)
                d2 = np.array(QuaternionY_data_smoothed)
                d2_feat = extract_features(d2)
                QuaternionZ_data = data_head['H-QuaternionZ']
                QuaternionZ_data_smoothed = smooth_data(QuaternionZ_data)
                d3 = np.array(QuaternionZ_data_smoothed)
                d3_feat = extract_features(d3)
                QuaternionW_data = data_head['H-QuaternionW']
                QuaternionW_data_smoothed = smooth_data(QuaternionW_data)
                d4 = np.array(QuaternionW_data_smoothed)
                d4_feat = extract_features(d4)

                Vector3X_data = data_head['H-Vector3X']
                Vector3X_data_smoothed = smooth_data(Vector3X_data)
                v1 = np.array(Vector3X_data_smoothed)
                v1_feat = extract_features(v1)
                Vector3Y_data = data_head['H-Vector3Y']
                Vector3Y_data_smoothed = smooth_data(Vector3Y_data)
                v2 = np.array(Vector3Y_data_smoothed)
                v2_feat = extract_features(v2)
                Vector3Z_data = data_head['H-Vector3Z']
                Vector3Z_data_smoothed = smooth_data(Vector3Z_data)
                v3 = np.array(Vector3Z_data_smoothed)
                v3_feat = extract_features(v3)

                # Eye points
                data_eye = pd.read_csv(
                    rotdir + "\GazeRaw_data_" + user + '-' + date + '-' + str(num + 1) + '.csv')
                QuaternionX_data = data_eye['L-QuaternionX']
                QuaternionX_data_smoothed = smooth_data(QuaternionX_data)
                d1_el = np.array(QuaternionX_data_smoothed)
                d1_el_feat = extract_features(d1_el)
                QuaternionY_data = data_eye['L-QuaternionY']
                QuaternionY_data_smoothed = smooth_data(QuaternionY_data)
                d2_el = np.array(QuaternionY_data_smoothed)
                d2_el_feat = extract_features(d2_el)
                QuaternionZ_data = data_eye['L-QuaternionZ']
                QuaternionZ_data_smoothed = smooth_data(QuaternionZ_data)
                d3_el = np.array(QuaternionZ_data_smoothed)
                d3_el_feat = extract_features(d3_el)
                QuaternionW_data = data_eye['L-QuaternionW']
                QuaternionW_data_smoothed = smooth_data(QuaternionW_data)
                d4_el = np.array(QuaternionW_data_smoothed)
                d4_el_feat = extract_features(d4_el)

                QuaternionX_data = data_eye['R-QuaternionX']
                QuaternionX_data_smoothed = smooth_data(QuaternionX_data)
                d1_er = np.array(QuaternionX_data_smoothed)
                d1_er_feat = extract_features(d1_er)
                QuaternionY_data = data_eye['R-QuaternionY']
                QuaternionY_data_smoothed = smooth_data(QuaternionY_data)
                d2_er = np.array(QuaternionY_data_smoothed)
                d2_er_feat = extract_features(d2_er)
                QuaternionZ_data = data_eye['R-QuaternionZ']
                QuaternionZ_data_smoothed = smooth_data(QuaternionZ_data)
                d3_er = np.array(QuaternionZ_data_smoothed)
                d3_er_feat = extract_features(d3_er)
                QuaternionW_data = data_eye['R-QuaternionW']
                QuaternionW_data_smoothed = smooth_data(QuaternionW_data)
                d4_er = np.array(QuaternionW_data_smoothed)
                d4_er_feat = extract_features(d4_er)

                # Head and eye points
                # difference_yaw_gaze_head_data = difference_yaw_gaze_head(user, date, num)

                # 利用所有特征：原始信号，切10段的特征，头眼yaw角度差
                # merged_array = np.concatenate([d1, d1_feat, d2, d2_feat, d3, d3_feat, d4, d4_feat, v1, v1_feat, v2, v2_feat, v3, v3_feat, d1_el, d1_el_feat, d2_el, d2_el_feat, d3_el, d3_el_feat, d4_el, d4_el_feat, d1_er, d1_er_feat, d2_er, d2_er_feat,  d3_er, d3_er_feat,  d4_er, d4_er_feat, difference_yaw_gaze_head_data[:features_sli]])

                # 利用特征：切10段的特征
                merged_array = np.concatenate(
                    [d1_feat, d2_feat, d3_feat, d4_feat, v1_feat, v2_feat, v3_feat, d1_el_feat, d2_el_feat, d3_el_feat,
                     d4_el_feat, d1_er_feat, d2_er_feat, d3_er_feat, d4_er_feat, ])

                # print(d1)
                # merged_array = np.concatenate(
                #     [v1])
                result_array = np.vstack([result_array, merged_array]) if result_array.size else merged_array

    return result_array

################################################################ knn 二分类
def knn4con_binary(authentications_per_person, user_names, dates, n_neighbors=3):
    # 生成示例数据
    # labels = np.repeat(np.arange(num_people), authentications_per_person)
    labels = np.array([0 if user == 'zs' else 1 for user in user_names for _ in range(authentications_per_person)])
    data = data_processing(authentications_per_person=authentications_per_person, user_names=user_names, dates=dates,
                           rotdir="E:\Desktop\data\VRAuth")
    # 打印示例数据形状
    # print("Data shape:", data.shape)
    # print("Labels shape:", labels.shape)

    # 数据标准化
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(data_scaled, labels, test_size=0.2)
    # print("testing shape:", X_test.shape)

    # 创建KNN模型
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)

    # 准确度
    accuracy = accuracy_score(y_test, y_pred)
    print("准确度:", accuracy)

    # 混淆矩阵
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("混淆矩阵:")
    print(conf_matrix)

    # 精确度、召回率、F1分数
    # precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    print("精确度:", precision)
    print("召回率:", recall)
    print("F1分数:", f1)

    # 分类报告
    class_report = classification_report(y_test, y_pred)
    print("分类报告:")
    print(class_report)

################################################################ knn 多分类
def knn4con_multi(authentications_per_person, user_names, dates, n_neighbors=3):
    # 生成示例数据
    num_people = len(user_names)
    labels = np.repeat(np.arange(num_people), authentications_per_person)
    # labels = np.array([0 if user == 'zs' else 1 for user in user_names for _ in range(authentications_per_person)])
    data = data_processing(authentications_per_person=authentications_per_person, user_names=user_names, dates=dates,
                           rotdir="E:\Desktop\data\VRAuth")
    # 打印示例数据形状
    # print("Data shape:", data.shape)
    # print("Labels shape:", labels.shape)

    # 数据标准化
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(data_scaled, labels, test_size=0.2)
    # print("testing shape:", X_test.shape)

    # 创建KNN模型
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)

    # 准确度
    accuracy = accuracy_score(y_test, y_pred)
    print("准确度:", accuracy)

    # 混淆矩阵
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("混淆矩阵:")
    print(conf_matrix)

    # 精确度、召回率、F1分数
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    # precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    print("精确度:", precision)
    print("召回率:", recall)
    print("F1分数:", f1)

    # 分类报告
    class_report = classification_report(y_test, y_pred)
    print("分类报告:")
    print(class_report)

################################################################ svm 二分类
def svm4con_binary(authentications_per_person, user_names, dates, kernel="linear", C=1):
    labels = np.array([0 if user == 'zs' else 1 for user in user_names for _ in range(authentications_per_person)])
    data = data_processing(authentications_per_person=authentications_per_person, user_names=user_names, dates=dates,
                           rotdir="E:\Desktop\data\VRAuth")
    # 打印示例数据形状
    # print("Data shape:", data.shape)
    # print("Labels shape:", labels.shape)

    # 数据标准化
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(data_scaled, labels, test_size=0.2)
    # print("testing shape:", X_test.shape)

    # 创建svm模型
    svm_model = SVC(kernel=kernel, C=C)
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
    # precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    print("精确度:", precision)
    print("召回率:", recall)
    print("F1分数:", f1)

    # 分类报告
    class_report = classification_report(y_test, y_pred)
    print("分类报告:")
    print(class_report)

################################################################ svm 多分类
def svm4con_multi(authentications_per_person, user_names, dates, kernel="linear", C=1):
    num_people = len(user_names)
    labels = np.repeat(np.arange(num_people), authentications_per_person)
    data = data_processing(authentications_per_person=authentications_per_person, user_names=user_names, dates=dates,
                           rotdir="E:\Desktop\data\VRAuth")
    # 打印示例数据形状
    # print("Data shape:", data.shape)
    # print("Labels shape:", labels.shape)

    # 数据标准化
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(data_scaled, labels, test_size=0.2)
    # print("testing shape:", X_test.shape)

    # 创建svm模型
    svm_model = SVC(kernel=kernel, C=C)
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
    # precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    print("精确度:", precision)
    print("召回率:", recall)
    print("F1分数:", f1)

    # 分类报告
    class_report = classification_report(y_test, y_pred)
    print("分类报告:")
    print(class_report)

################################################################ svm kfold 二分类
def svm4con_binary_kfolds(authentications_per_person, user_names, dates, kernel="linear", C=1, n_splits=3):
    labels = np.array([0 if user == 'zs' else 1 for user in user_names for _ in range(authentications_per_person)])
    data = data_processing(authentications_per_person=authentications_per_person, user_names=user_names, dates=dates,
                           rotdir="E:\Desktop\data\VRAuth")
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    for label in np.unique(labels):
        count = np.sum(labels == label)
        # print(f"Class {label}: {count} samples")
    # 设置交叉验证的折叠数
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True)

    # 初始化准确性列表，用于存储每个折叠的模型准确性
    accuracies = []
    precisions = []
    recalls = []
    f1s = []

    for train_index, test_index in kf.split(data_scaled, labels):
        X_train, X_test = data_scaled[train_index], data_scaled[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        # 创建SVM模型
        svm_model = SVC(kernel=kernel, C=C)

        # 模型训练
        svm_model.fit(X_train, y_train)

        # 模型预测
        y_pred = svm_model.predict(X_test)

        # 计算准确性
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    # # 打印每个折叠的准确性
    # for i, acc in enumerate(accuracies):
    #     print(f"Fold {i + 1} Accuracy: {acc}")

    # 打印平均准确性 精确 召回 f1
    average_accuracy = np.mean(accuracies)
    # average_precision = np.mean(precisions)
    # average_recalls = np.mean(recalls)
    # average_f1s = np.mean(f1s)
    print("Average Accuracy:", average_accuracy, "\nprecision:", precisions, "\nrecalls:", recalls, "\nf1s:", f1s)

################################################################ svm kfold 多分类
def svm4con_multi_kfolds(authentications_per_person, user_names, dates, kernel="linear", C=1, n_splits=3):
    num_people = len(user_names)
    labels = np.repeat(np.arange(num_people), authentications_per_person)
    data = data_processing(authentications_per_person=authentications_per_person, user_names=user_names, dates=dates,
                           rotdir="E:\Desktop\data\VRAuth")
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    for label in np.unique(labels):
        count = np.sum(labels == label)
        # print(f"Class {label}: {count} samples")
    # 设置交叉验证的折叠数
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True)

    # 初始化准确性列表，用于存储每个折叠的模型准确性
    accuracies = []
    precisions = []
    recalls = []
    f1s = []

    for train_index, test_index in kf.split(data_scaled, labels):
        X_train, X_test = data_scaled[train_index], data_scaled[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        # 创建SVM模型
        svm_model = SVC(kernel=kernel, C=C)

        # 模型训练
        svm_model.fit(X_train, y_train)

        # 模型预测
        y_pred = svm_model.predict(X_test)

        # 计算准确性
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    # # 打印每个折叠的准确性
    # for i, acc in enumerate(accuracies):
    #     print(f"Fold {i + 1} Accuracy: {acc}")
    # 打印平均准确性 精确 召回 f1
    average_accuracy = np.mean(accuracies)
    # average_precision = np.mean(precisions)
    # average_recalls = np.mean(recalls)
    # average_f1s = np.mean(f1s)
    print("Average Accuracy:", average_accuracy, "\nprecision:", precisions, "\nrecalls:", recalls, "\nf1s:", f1s)


################################################################ main
def main():
    # 每个人采集次数
    authentications_per_person = 6

    user_names = ['zs', 'zjr', 'gj', 'pyj']

    dates = ['1118']

    print("--------knn_binary------------")
    knn4con_binary(authentications_per_person=authentications_per_person, user_names=user_names, dates=dates)

    print("---------knn_multi------------")
    knn4con_multi(authentications_per_person=authentications_per_person, user_names=user_names, dates=dates)

    print("---------svm_binary------------")
    svm4con_binary(authentications_per_person=authentications_per_person, user_names=user_names, dates=dates)

    print("---------svm_multi------------")
    svm4con_multi(authentications_per_person=authentications_per_person, user_names=user_names, dates=dates)

    print("----------svm_binary_kfold------------")
    svm4con_binary_kfolds(authentications_per_person=authentications_per_person, user_names=user_names, dates=dates)

    print("-----------svm_multi------------")
    svm4con_multi_kfolds(authentications_per_person=authentications_per_person, user_names=user_names, dates=dates)

if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, classification_report
import os, json
from data_preprocess import difference_gaze_head
from sklearn.svm import SVC


def smooth_data(arr, window_parameter=9, polyorder_parameter=2):
    arr_smoothed = savgol_filter(arr, window_length=window_parameter, polyorder=polyorder_parameter)
    return arr_smoothed

def read_data_name_from_json(filepath = "src/data.json"):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    result_list = [f"{item['studytype']}-{item['names']}-{item['date']}" for item in data['data']]
    return result_list

def add_small_noise(sequence, noise_level=0.01):
    noise = np.random.normal(0, noise_level, len(sequence))
    augmented_sequence = sequence + noise
    return augmented_sequence


def extract_features(sequence):  # 把序列切成十段，每段取均值、最大值、最小值、方差，共40个特征，返回一个拼接的一维数组
    # 计算每个子序列的基本长度和额外长度
    n = len(sequence)
    # print("length" + str(n))
    slice_num = 10
    sub_seq_length = n // slice_num if n % slice_num == 0 else n // slice_num + 1# 向上取整
    remainder = sub_seq_length - (n // slice_num + 1) * slice_num + n # 处理最后一段未填充满

    # 初始化特征数组
    features = []
    features_mean = []       # 均值
    features_max = []        # 最大值
    features_min = []        # 最小值
    features_var = []        # 方差
    features_median = []     # 中位数
    features_rms = []        # 均方根值
    features_std = []        # 标准差
    features_mad = []        # 平均绝对偏差
    features_kurtosis = []   # 峰度
    features_skewness = []   # 偏度
    features_iqr = []        # 四分位数范围
    # features_roughness = []  # 粗糙度，需要具体定义
    # features_sharpness = []  # 锋利度，需要具体定义
    features_mc = []         # 均值穿越次数
    features_wamp = []       # Willison幅度
    features_ssc = []        # 坡度符号变化次数
    start = 0

    # 对每个子序列进行迭代
    for i in range(slice_num):
        # 调整子序列长度
        end = start + sub_seq_length if i < slice_num - 1 else start + (remainder if remainder > 0 else sub_seq_length)
        sub_seq = sequence[start:end]

        # 计算特征
        mean = np.mean(sub_seq)                         # 计算均值
        max_value = np.max(sub_seq)                     # 计算最大值
        min_value = np.min(sub_seq)                     # 计算最小值
        variance = np.var(sub_seq)                      # 计算方差
        median = np.median(sub_seq)                     # 计算中位数
        rms = np.sqrt(np.mean(np.square(sub_seq)))      # 计算均方根
        std_dev = np.std(sub_seq)                       # 计算标准差
        mad = np.mean(np.abs(sub_seq - np.mean(sub_seq))) # 计算平均绝对偏差
        kurt = kurtosis(sub_seq)                        # 计算峰度
        skewness = skew(sub_seq)                        # 计算偏度
        q75, q25 = np.percentile(sub_seq, [75, 25])
        iqr = q75 - q25                                 # 计算四分位数范围
        mc = np.sum(np.sign(sub_seq[:-1]) != np.sign(sub_seq[1:])) / len(sub_seq) # 计算均值穿越次数
        threshold = 0.1  # 根据需要设置阈值
        wamp = np.sum(np.abs(np.diff(sub_seq)) > threshold) # 计算Willison幅度
        ssc = np.sum(np.diff(np.sign(np.diff(sub_seq))) != 0) # 计算坡度符号变化次数

        # 添加到特征数组
        features_mean.append(mean)
        features_max.append(max_value)
        features_min.append(min_value)
        features_var.append(variance)
        features_median.append(median)
        features_rms.append(rms)
        features_std.append(std_dev)
        features_mad.append(mad)
        features_kurtosis.append(kurt)
        features_skewness.append(skewness)
        features_iqr.append(iqr)
        # features_roughness.append(roughness)  # 根据定义实现
        # features_sharpness.append(sharpness)  # 根据定义实现
        features_mc.append(mc)
        features_wamp.append(wamp)
        features_ssc.append(ssc)

        # 更新起始位置
        start = end


    # return np.concatenate([features_max])

    return np.concatenate([features_mean, features_max, features_min, features_var,
                           features_median, features_rms, features_std, features_mad,
                           features_kurtosis, features_skewness, features_iqr,
                           features_mc, features_wamp, features_ssc])


def data_scaled_and_label(studytype_users_dates, rotdir=None, model="", size_num_study1= 6, pin_num = 4, authentications_per_person=6, positive_label=None):
    import itertools
    def map_names_to_numbers(names):
        name_to_number = {}
        number_list = []
        counter = 1
        for name in names:
            if name not in name_to_number:
                name_to_number[name] = counter
                counter += 1
            number_list.append(name_to_number[name])
        return number_list

    # 特征
    result_array = np.array([])
    if studytype_users_dates.split('-')[0] == 'study1':
        authentications_per_person = 2
        size_pin_num_pair = itertools.product(range(1, size_num_study1+1), range(1, pin_num+1))
        for member in studytype_users_dates:
        # 特征:
            for size, pin in size_pin_num_pair:
                for num in range(authentications_per_person):
                # Head
                    data_head = pd.read_csv(
                        rotdir + f"Head_data_{member}-{str(size)}-{str(pin)}-{str(num+1)}.csv")
                    QuaternionX_data = data_head['H-QuaternionX']
                    QuaternionX_data = QuaternionX_data - np.mean(QuaternionX_data[0:5])
                    QuaternionX_data_smoothed = smooth_data(QuaternionX_data)
                    d1 = np.array(QuaternionX_data_smoothed)
                    d1_feat = extract_features(d1)
                    QuaternionY_data = data_head['H-QuaternionY']
                    QuaternionY_data = QuaternionY_data - np.mean(QuaternionY_data[0:5])
                    QuaternionY_data_smoothed = smooth_data(QuaternionY_data)
                    d2 = np.array(QuaternionY_data_smoothed)
                    d2_feat = extract_features(d2)
                    QuaternionZ_data = data_head['H-QuaternionZ']
                    QuaternionZ_data = QuaternionZ_data - np.mean(QuaternionZ_data[0:5])
                    QuaternionZ_data_smoothed = smooth_data(QuaternionZ_data)
                    d3 = np.array(QuaternionZ_data_smoothed)
                    d3_feat = extract_features(d3)
                    QuaternionW_data = data_head['H-QuaternionW']
                    QuaternionW_data = QuaternionW_data - np.mean(QuaternionW_data[0:5])
                    QuaternionW_data_smoothed = smooth_data(QuaternionW_data)
                    d4 = np.array(QuaternionW_data_smoothed)
                    d4_feat = extract_features(d4)

                    Vector3X_data = data_head['H-Vector3X']
                    Vector3X_data = Vector3X_data - np.mean(Vector3X_data[0:5])
                    Vector3X_data_smoothed = smooth_data(Vector3X_data)
                    v1 = np.array(Vector3X_data_smoothed)
                    v1_feat = extract_features(v1)
                    Vector3Y_data = data_head['H-Vector3Y']
                    Vector3Y_data = Vector3Y_data - np.mean(Vector3Y_data[0:5])
                    Vector3Y_data_smoothed = smooth_data(Vector3Y_data)
                    v2 = np.array(Vector3Y_data_smoothed)
                    v2_feat = extract_features(v2)
                    Vector3Z_data = data_head['H-Vector3Z']
                    Vector3Z_data = Vector3Z_data - np.mean(Vector3Z_data[0:5])
                    Vector3Z_data_smoothed = smooth_data(Vector3Z_data)
                    v3 = np.array(Vector3Z_data_smoothed)
                    v3_feat = extract_features(v3)

                    # Eye points
                    data_eye = pd.read_csv(
                        rotdir + f"\GazeRaw_data_{member}-{str(size)}-{str(pin)}-{str(num+1)}.csv")
                    QuaternionX_data = data_eye['L-QuaternionX']
                    QuaternionX_data = QuaternionX_data - np.mean(QuaternionX_data[0:5])
                    QuaternionX_data_smoothed = smooth_data(QuaternionX_data)
                    d1_el = np.array(QuaternionX_data_smoothed)
                    d1_el_feat = extract_features(d1_el)
                    QuaternionY_data = data_eye['L-QuaternionY']
                    QuaternionY_data = QuaternionY_data - np.mean(QuaternionY_data[0:5])
                    QuaternionY_data_smoothed = smooth_data(QuaternionY_data)
                    d2_el = np.array(QuaternionY_data_smoothed)
                    d2_el_feat = extract_features(d2_el)
                    QuaternionZ_data = data_eye['L-QuaternionZ']
                    QuaternionZ_data = QuaternionZ_data - np.mean(QuaternionZ_data[0:5])
                    QuaternionZ_data_smoothed = smooth_data(QuaternionZ_data)
                    d3_el = np.array(QuaternionZ_data_smoothed)
                    d3_el_feat = extract_features(d3_el)
                    QuaternionW_data = data_eye['L-QuaternionW']
                    QuaternionW_data = QuaternionW_data - np.mean(QuaternionW_data[0:5])
                    QuaternionW_data_smoothed = smooth_data(QuaternionW_data)
                    d4_el = np.array(QuaternionW_data_smoothed)
                    d4_el_feat = extract_features(d4_el)

                    QuaternionX_data = data_eye['R-QuaternionX']
                    QuaternionX_data = QuaternionX_data - np.mean(QuaternionX_data[0:5])
                    QuaternionX_data_smoothed = smooth_data(QuaternionX_data)
                    d1_er = np.array(QuaternionX_data_smoothed)
                    d1_er_feat = extract_features(d1_er)
                    QuaternionY_data = data_eye['R-QuaternionY']
                    QuaternionY_data = QuaternionY_data - np.mean(QuaternionY_data[0:5])
                    QuaternionY_data_smoothed = smooth_data(QuaternionY_data)
                    d2_er = np.array(QuaternionY_data_smoothed)
                    d2_er_feat = extract_features(d2_er)
                    QuaternionZ_data = data_eye['R-QuaternionZ']
                    QuaternionZ_data = QuaternionZ_data - np.mean(QuaternionZ_data[0:5])
                    QuaternionZ_data_smoothed = smooth_data(QuaternionZ_data)
                    d3_er = np.array(QuaternionZ_data_smoothed)
                    d3_er_feat = extract_features(d3_er)
                    QuaternionW_data = data_eye['R-QuaternionW']
                    QuaternionW_data = QuaternionW_data - np.mean(QuaternionW_data[0:5])
                    QuaternionW_data_smoothed = smooth_data(QuaternionW_data)
                    d4_er = np.array(QuaternionW_data_smoothed)
                    d4_er_feat = extract_features(d4_er)

                    # Head and eye points
                    diff_yaw_data = difference_gaze_head(member, size, pin, num+1)
                    diff_yaw_smooth = smooth_data(diff_yaw_data, window_parameter=9)
                    dy_el_feat = extract_features(np.array(diff_yaw_smooth))
                    diff_pitch_data = difference_gaze_head(member, size, pin, num+1, eye='L', angle='Pitch')
                    diff_pitch_smooth = smooth_data(diff_pitch_data, window_parameter=9)
                    dp_el_feat = extract_features(np.array(diff_pitch_smooth))
                    diff_roll_data = difference_gaze_head(member, size, pin, num+1, eye='L', angle='Roll')
                    diff_roll_smooth = smooth_data(diff_roll_data, window_parameter=9)
                    dr_el_feat = extract_features(np.array(diff_roll_smooth))

                    if model == 'head':
                        merged_array = np.concatenate(
                            [d1_feat, d2_feat, d3_feat, d4_feat, v1_feat, v2_feat, v3_feat])
                    # 利用特征：切10段的特征
                    elif model == "eye":
                        merged_array = np.concatenate(
                            [d1_el_feat, d2_el_feat, d3_el_feat,
                                d4_el_feat, d1_er_feat, d2_er_feat, d3_er_feat, d4_er_feat, ])
                    elif model == "head+eye":
                        # 利用特征：切10段的特征
                        merged_array = np.concatenate(
                            [d1_feat, d2_feat, d3_feat, d4_feat, v1_feat, v2_feat, v3_feat, d1_el_feat, d2_el_feat,
                                d3_el_feat,
                                d4_el_feat, d1_er_feat, d2_er_feat, d3_er_feat, d4_er_feat, ])
                    elif model == "diff":
                        # 利用特征：切10段的特征
                        merged_array = np.concatenate([dy_el_feat, dp_el_feat, dr_el_feat]
                        )
                    elif model == "eye+diff":
                        merged_array = np.concatenate([dy_el_feat, dp_el_feat,
                                                        dr_el_feat, d1_el_feat, d2_el_feat, d3_el_feat,
                                d4_el_feat, d1_er_feat, d2_er_feat, d3_er_feat, d4_er_feat]
                        )
                    else:
                        merged_array = np.concatenate(
                            [d1_feat, d2_feat, d3_feat, d4_feat, v1_feat, v2_feat, v3_feat, d1_el_feat, d2_el_feat,
                                d3_el_feat,
                                d4_el_feat, d1_er_feat, d2_er_feat, d3_er_feat, d4_er_feat,
                                dy_el_feat, dp_el_feat, dr_el_feat])
                    # if num == 1:
                    #     print("user" + user + "data" + str(merged_array))
                    result_array = np.vstack([result_array, merged_array]) if result_array.size else merged_array
    
    scaler = StandardScaler()
    users = [x.split('-')[1] for x in studytype_users_dates]
    num_people = len(map_names_to_numbers(users))
    labels = print(np.repeat(map_names_to_numbers(users), authentications_per_person))
    binary_labels = np.array([1 if user in positive_label else 0 for user in users for _ in range(authentications_per_person)])
    return scaler.fit_transform(result_array), labels, binary_labels


################################################################ knn 二分类
def knn4con_binary( model, n_neighbors=3,
                   latter_auth_per_person=0, latter_user_names=None, latter_dates=None, latter_positive_label=None,data_scaled=None, binary_labels=None):
    # 生成示例数据
    # labels = np.repeat(np.arange(num_people), authentications_per_person)

    # 打印示例数据形状
    # print("Data shape:", data.shape)
    # print("Labels shape:", labels.shape)
    # print("authentications_per_person", authentications_per_person, 'user_names', user_names)

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(data_scaled, binary_labels, test_size=0.2)
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

    # # 精确度、召回率、F1分数
    # # precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    # precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=True)
    # print("精确度:", precision)
    # print("召回率:", recall)
    # print("F1分数:", f1)
    #
    # # 从混淆矩阵中提取真正例（True Positives）、假正例（False Positives）、真负例（True Negatives）、假负例（False Negatives）
    # true_positive = conf_matrix[1, 1]
    # false_positive = conf_matrix[0, 1]
    # true_negative = conf_matrix[0, 0]
    # false_negative = conf_matrix[1, 0]
    #
    # # 计算 FAR 和 FRR
    # far = false_positive / (false_positive + true_negative)
    # frr = false_negative / (false_negative + true_positive)
    #
    # # 打印结果
    # print(f"FAR: {far:.4f}")
    # print(f"FRR: {frr:.4f}")

    # if latter_auth_per_person != 0:
    #     latter_labels = np.array(
    #         [1 if latter_user in latter_positive_label else 0 for latter_user in latter_user_names for _ in range(latter_auth_per_person)])
    #     latter_data = data_processing(authentications_per_person=latter_auth_per_person, user_names=latter_user_names, dates=latter_dates,
    #                                 rotdir="E:\Desktop\data\VRAuth", model=model)

    #     latter_data_scaled = scaler.transform(latter_data)
    #     latter_y_pred = knn_model.predict(latter_data_scaled)
    #     print(latter_y_pred, latter_labels)

    #     latter_accuracy = accuracy_score(latter_y_pred, latter_labels)
    #     print('随时间推移的准确率', latter_accuracy)


################################################################ knn 多分类
def knn4con_multi(model, n_neighbors=3, data_scaled=None, labels=None):
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

    # # 分类报告
    # class_report = classification_report(y_test, y_pred)
    # print("分类报告:")
    # print(class_report)


################################################################ svm 二分类
def svm4con_binary(authentications_per_person, user_names, dates, model, kernel="linear", C=1, gamma=0.02, positive_label=None,
                   latter_auth_per_person=0, latter_user_names=None, latter_dates=None, latter_positive_label=None, data_scaled=None, binary_labels=None):
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(data_scaled, binary_labels, test_size=0.2)
    # print("testing shape:", X_test.shape)

    # 创建svm模型
    svm_model = SVC(kernel=kernel, C=C, gamma=gamma,)
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

    # 从混淆矩阵中提取真正例（True Positives）、假正例（False Positives）、真负例（True Negatives）、假负例（False Negatives）
    true_positive = conf_matrix[1, 1]
    false_positive = conf_matrix[0, 1]
    true_negative = conf_matrix[0, 0]
    false_negative = conf_matrix[1, 0]

    # 计算 FAR 和 FRR
    far = false_positive / (false_positive + true_negative)
    frr = false_negative / (false_negative + true_positive)

    # 打印结果
    print(f"FAR: {far:.4f}")
    print(f"FRR: {frr:.4f}")

    # if latter_auth_per_person != 0:
    #     latter_labels = np.array(
    #         [1 if latter_user in latter_positive_label else 0 for latter_user in latter_user_names for _ in
    #          range(latter_auth_per_person)])
    #     latter_data = data_processing(authentications_per_person=latter_auth_per_person, user_names=latter_user_names,
    #                                   dates=latter_dates,
    #                                   rotdir="E:\Desktop\data\VRAuth", model=model)

    #     latter_data_scaled = scaler.transform(latter_data)
    #     latter_y_pred = svm_model.predict(latter_data_scaled)
    #     print(latter_y_pred, latter_labels)

    #     latter_accuracy = accuracy_score(latter_y_pred, latter_labels)
    #     print('随时间推移的准确率', latter_accuracy)


################################################################ svm 多分类
def svm4con_multi(authentications_per_person, user_names, dates, model, kernel="linear", C=1, gamma=0.02,
                  latter_auth_per_person=0, latter_user_names=None, latter_dates=None, data_scaled=None, labels=None):

    num_people = len(user_names)
    labels = np.repeat(np.arange(num_people), authentications_per_person)
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(data_scaled, labels, test_size=0.2)

    # 创建svm模型
    svm_model = SVC(kernel=kernel, C=C, gamma=gamma)
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


################################################################ knn kfold 二分类
def knn4con_binary_kfolds(n_neighbors=3, n_splits=5, data_scaled=None, binary_labels=None, model=""):


    for label in np.unique(binary_labels):
        count = np.sum(binary_labels == label)
        # print(f"Class {label}: {count} samples")
    # 设置交叉验证的折叠数
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True)

    # 初始化准确性列表，用于存储每个折叠的模型准确性
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    fars = []
    frrs = []

    for train_index, test_index in kf.split(data_scaled, binary_labels):
        X_train, X_test = data_scaled[train_index], data_scaled[test_index]
        y_train, y_test = binary_labels[train_index], binary_labels[test_index]

        knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn_model.fit(X_train, y_train)
        y_pred = knn_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=1)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        conf_matrix = confusion_matrix(y_test, y_pred)
        true_positive = conf_matrix[1, 1]
        false_positive = conf_matrix[0, 1]
        true_negative = conf_matrix[0, 0]
        false_negative = conf_matrix[1, 0]

        # 计算 FAR 和 FRR
        far = false_positive / (false_positive + true_negative)
        frr = false_negative / (false_negative + true_positive)

        fars.append(far)
        frrs.append(frr)

        # # 打印每个折叠的准确性
        # for i, acc in enumerate(accuracies):
        #     print(f"Fold {i + 1} Accuracy: {acc}")

        # 打印平均准确性 精确 召回 f1
    average_accuracy = np.mean(accuracies)
    average_precision = np.mean(precisions)
    average_recalls = np.mean(recalls)
    average_f1s = np.mean(f1s)
    average_fars = np.mean(fars)
    average_frrs = np.mean(frrs)
    print("Average Accuracy:", average_accuracy, "\nprecision:", average_precision, "\nrecalls:", average_recalls,
          "\nf1s:", average_f1s, "\nfars:", average_fars, "\nfrrs:", average_frrs)


################################################################ knn kfold 多分类
def knn4con_multi_kfolds(model, n_neighbors=3, n_splits=5, data_scaled=None, labels=None):
    # 生成示例数据
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

        knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn_model.fit(X_train, y_train)
        y_pred = knn_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=1)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    average_accuracy = np.mean(accuracies)
    average_precision = np.mean(precisions)
    average_recalls = np.mean(recalls)
    average_f1s = np.mean(f1s)

    print("Average Accuracy:", average_accuracy, "\nprecision:", average_precision, "\nrecalls:", average_recalls,
          "\nf1s:", average_f1s)


################################################################ svm kfold 二分类
def svm4con_binary_kfolds(kernel="linear", C=1, gamma = 0.02, n_splits=3, data_scaled=None, binary_labels=None):

    for label in np.unique(binary_labels):
        count = np.sum(binary_labels == label)
        # print(f"Class {label}: {count} samples")
    # 设置交叉验证的折叠数
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True)

    # 初始化准确性列表，用于存储每个折叠的模型准确性
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    fars = []
    frrs = []

    for train_index, test_index in kf.split(data_scaled, binary_labels):
        X_train, X_test = data_scaled[train_index], data_scaled[test_index]
        y_train, y_test = binary_labels[train_index], binary_labels[test_index]

        # 创建SVM模型
        svm_model = SVC(kernel=kernel, C=C, gamma=gamma)

        # 模型训练
        svm_model.fit(X_train, y_train)

        # 模型预测
        y_pred = svm_model.predict(X_test)

        # 计算准确性
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=1)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        conf_matrix = confusion_matrix(y_test, y_pred)
        true_positive = conf_matrix[1, 1]
        false_positive = conf_matrix[0, 1]
        true_negative = conf_matrix[0, 0]
        false_negative = conf_matrix[1, 0]

        # 计算 FAR 和 FRR
        far = false_positive / (false_positive + true_negative)
        frr = false_negative / (false_negative + true_positive)

        fars.append(far)
        frrs.append(frr)

    # # 打印每个折叠的准确性
    # for i, acc in enumerate(accuracies):
    #     print(f"Fold {i + 1} Accuracy: {acc}")

    # 打印平均准确性 精确 召回 f1
    average_accuracy = np.mean(accuracies)
    average_precision = np.mean(precisions)
    average_recalls = np.mean(recalls)
    average_f1s = np.mean(f1s)
    average_fars = np.mean(fars)
    average_frrs = np.mean(frrs)
    print("Average Accuracy:", average_accuracy, "\nprecision:", average_precision, "\nrecalls:", average_recalls,
          "\nf1s:", average_f1s, "\nfars:", average_fars, "\nfrrs:", average_frrs)


################################################################ svm kfold 多分类
def svm4con_multi_kfolds(model, kernel="linear", C=1, gamma = 0.02, n_splits=3, data_scaled=None, labels=None):
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
        svm_model = SVC(kernel=kernel, C=C, gamma=gamma)

        # 模型训练
        svm_model.fit(X_train, y_train)

        # 模型预测
        y_pred = svm_model.predict(X_test)

        # 计算准确性
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=1)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    average_accuracy = np.mean(accuracies)
    average_precision = np.mean(precisions)
    average_recalls = np.mean(recalls)
    average_f1s = np.mean(f1s)

    print("Average Accuracy:", average_accuracy, "\nprecision:", average_precision, "\nrecalls:", average_recalls,
          "\nf1s:", average_f1s)


################################################################ main
def main():
    ################################ train和test部分
    # 每个人采集次数
    # authentications_per_person = 6

    # user_names = ["m-zhao", 'mzs', "m-zjr", 'mhyr', 'mljq', 'mgx', 'mlwk', ]
    # user_names = ['m-zjr', 'm-zhao']
    # scene = ""

    

    # for i in range(len(user_names)):
    #     user_names[i] = scene + user_names[i]

    positive_label = [1,2]  # 正样本
    data_scaled, labels, binery_labels = data_scaled_and_label(authentications_per_person=2, rotdir = "data/", positive_label=positive_label, model="head+eye+diff", studytype_users_dates=read_data_name_from_json(), size_num_study1=6, pin_num=4)

    model = ''  # model

    n_split = 5  # k fold

    

    kernel = "linear"

    print("---------knn_binary_kfold------------")
    knn4con_binary_kfolds(data_scaled=data_scaled, binary_labels=binery_labels,
                          model=model, positive_label=positive_label, n_splits=n_split)

    print("---------knn_multi_kfold------------")
    knn4con_multi_kfolds(data_scaled=data_scaled, labels=labels,
                         model=model,
                         n_splits=n_split)

    print("----------svm_binary_kfold------------")
    svm4con_binary_kfolds(data_scaled=data_scaled, binary_labels=binery_labels,
                          model=model, positive_label=positive_label, n_splits=n_split, kernel=kernel)

    print("-----------svm_multi_kfold------------")
    svm4con_multi_kfolds(data_scaled=data_scaled, labels=labels,
                         model=model,
                         n_splits=n_split, kernel=kernel)

    # ################################ 随时间推移重新检验部分
    # latter_auth_per_person = 6

    # latter_user_names = ['m-zhao']
    # latter_scene = ''

    # for i in range(len(latter_user_names)):
    #     latter_user_names[i] = latter_scene + latter_user_names[i]

    # latter_dates = ['1216']

    # latter_positive_label = [latter_user_names[0]]

    # print("--------knn_binary------------")
    # knn4con_binary(authentications_per_person=authentications_per_person, user_names=user_names, dates=dates,
    #                model=model, positive_label=positive_label, latter_auth_per_person=latter_auth_per_person, latter_user_names=latter_user_names,
    #                latter_dates=latter_dates, latter_positive_label=latter_positive_label)

    # print("---------knn_multi------------")
    # knn4con_multi(authentications_per_person=authentications_per_person, user_names=user_names, dates=dates, model=model, )

    # print("---------svm_binary------------")
    # svm4con_binary(authentications_per_person=authentications_per_person, user_names=user_names, dates=dates,
    #                model=model, positive_label=positive_label, latter_auth_per_person=latter_auth_per_person, latter_user_names=latter_user_names,
    #                latter_dates=latter_dates, latter_positive_label=latter_positive_label)

    # print("---------svm_multi------------")
    # svm4con_multi(authentications_per_person=authentications_per_person, user_names=user_names, dates=dates, model=model,
    #               latter_auth_per_person=latter_auth_per_person, latter_user_names=latter_user_names,
    #               latter_dates=latter_dates, data_scaled=data_scaled, labels=labels
    #               )



if __name__ == "__main__":
    main()
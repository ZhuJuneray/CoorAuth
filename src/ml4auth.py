import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, classification_report
import os, json, re, sys
from data_preprocess import difference_gaze_head, add_noise, feature_process_quaternion, merged_array_generator
from sklearn.svm import SVC


def smooth_data(arr, window_parameter=9, polyorder_parameter=2):
    arr_smoothed = savgol_filter(arr, window_length=window_parameter, polyorder=polyorder_parameter)
    return arr_smoothed


def read_data_name_from_json(filepath="D:\pycharm\srt_vr_auth\src\data.json"):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    data_list = [f"{item['studytype']}_{item['names']}_{item['date']}_{item['num_range']}" for item in data['data']]
    latter_auth_list = [f"{item['studytype']}_{item['names']}_{item['date']}_{item['num_range']}" for item in
                        data['latter_auth']]
    return data_list, latter_auth_list


def add_small_noise(sequence, noise_level=0.01):
    noise = np.random.normal(0, noise_level, len(sequence))
    augmented_sequence = sequence + noise
    return augmented_sequence


def data_scaled_and_label(studytype_users_dates_range, rotdir=None, model="", size_list=None,
                          pin_list=[1, 2, 3, 4], default_authentications_per_person=6,
                          positive_label=None, noise_level=0.1, scaler_origin=None, scaler_augment=None):  # 返回scaled后的原始数据和标签，scaled后的增强后的数据和标签
    if pin_list is None:
        pin_list = [1, 2, 3, 4]
    import itertools

    studytype_user_date_size_pin_num_pair = []  # studytype, user, date, size, pin, num 的所有排列组合，用于数据增强时循环增强所有正标签里的数据
    result_array = np.array([])
    studytype = studytype_users_dates_range[0].split('_')[0]  # studytype只有一种
    users = [x.split('_')[1] for x in studytype_users_dates_range]
    dates = [x.split('_')[2] for x in studytype_users_dates_range]
    _ranges = [x.split('_')[3] for x in studytype_users_dates_range]
    labels = []
    binary_labels = []
    for member in studytype_users_dates_range:
        # studytype = member.split('_')[0]
        user = member.split('_')[1]
        date = member.split('_')[2]
        num_range = member.split('_')[3]
        range_start = int(num_range.split('-')[0]) if num_range else 1
        range_end = int(num_range.split('-')[1]) if num_range else default_authentications_per_person

        studytype_user_date_size_pin_num_pair.extend([x for x in
                                                      itertools.product([studytype], [user], [date], size_list,
                                                                        pin_list, range(range_start, range_end + 1))])

        # 标签:
        labels.extend(np.repeat(user, len(size_list) * len(pin_list) * (range_end - range_start + 1)))
        binary_labels.extend([1 if user in positive_label else 0 for _ in
                              range(len(size_list) * len(pin_list) * ((range_end - range_start + 1)))])

        # 特征:
        for size in size_list:
            for pin in pin_list:
                for num in range(range_start, range_end + 1):
                    # print(f"num:{num}")
                    merged_array = merged_array_generator(member=member, rotdir=rotdir, model=model, size=size, pin=pin,
                                                          num=num,
                                                          noise_flag=False)  # 返回该member, size, pin, num的特征, 返回的是可以直接用于机器学习的一维X向量
                    result_array = np.vstack(
                        [result_array, merged_array]) if result_array.size else merged_array  # 将所有特征堆叠起来，每一行是一个特征

    # scaler的fit, 如果未传入scaler则创建一个) 1231update 弃用
    # if scaler_origin is None:
    #     scaler_origin = StandardScaler()
    #     scaler_origin.fit(result_array)
    # scaled_data = scaler_origin.transform(result_array)  # 标准化数据
    scaled_data = result_array
    # num_people = len(map_names_to_numbers(users))
    # if studytype_users_dates[0].split('-')[0] == 'study1':
    #     labels = np.repeat(users, authentications_per_person*len(size_num_study1)*len(pin_num))
    #     binary_labels = np.array([1 if user in positive_label else 0 for user in users for _ in range(authentications_per_person*len(size_num_study1)*len(pin_num))])
    # if studytype_users_dates[0].split('-')[0] == 'study2':
    #     labels = np.repeat(users, authentications_per_person*len(size_num_study2)*len(pin_num))
    #     binary_labels = np.array([1 if user in positive_label else 0 for user in users for _ in range(authentications_per_person*len(size_num_study2)*len(pin_num))])

    # 识别正类样本
    positive_indices = np.where(binary_labels == 1)[0]
    positive_features = result_array[positive_indices]

    # 确定增强的正样本数量以达到大约50%的正样本比例
    total_samples_needed = len(binary_labels)  # 总样本数
    positive_samples_needed = int(total_samples_needed) - 2 * len(positive_indices)  # 需要增强的正样本数

    # 如果需要增强的样本数为负数或零，则不执行任何操作
    if positive_samples_needed > 0:
        # 选择正样本进行复制和添加噪声
        users_to_copy = np.random.choice(positive_label, size=positive_samples_needed, replace=True)
        loop_num = 0
        i = 0
        j = 0
        positive_features_to_augment = np.array([])
        # studytype user date size pin num

        while loop_num < positive_samples_needed:
            user_to_copy = users_to_copy[loop_num]
            studytype_user_date_size_pin_num_pair_to_copy = [x for x in studytype_user_date_size_pin_num_pair if
                                                             x[1] == user_to_copy]  # user_to_copy的所有组合
            studytype_to_copy = studytype_user_date_size_pin_num_pair_to_copy[j][0]
            date_to_copy = studytype_user_date_size_pin_num_pair_to_copy[j][2]
            size_to_copy = studytype_user_date_size_pin_num_pair_to_copy[j][3]
            pin_to_copy = studytype_user_date_size_pin_num_pair_to_copy[j][4]
            num_to_copy = studytype_user_date_size_pin_num_pair_to_copy[j][5]
            # print(f"studytype_user_date_size_pin_num_pair_to_copy:{studytype_user_date_size_pin_num_pair_to_copy}")
            # pattern = '^'+studytype+f'-{user_to_copy}-.*$'
            # user_size_pin_num_to_copy = [x for x in user_size_pin_num_pair if x[0] == user_to_copy]
            member_to_copy = f"{studytype_to_copy}_{user_to_copy}_{date_to_copy}"  # 用于merged_array_generator的member参数
            merged_array_augmented = merged_array_generator(member=member_to_copy, rotdir=rotdir, model=model,
                                                            size=size_to_copy, pin=pin_to_copy, num=num_to_copy,
                                                            noise_flag=True, noise_level=noise_level)
            positive_features_to_augment = np.vstack([positive_features_to_augment,
                                                      merged_array_augmented]) if positive_features_to_augment.size else merged_array_augmented

            # if k == len(member_to_copy)-1 and j == len(size_pin_num_pair)-1:
            #     i= (i+1)%len(users_to_copy) #选user复制
            #     j = (j+1)%len(size_pin_num_pair)
            #     k = (k+1)%len(member_to_copy)
            # elif j == len(size_pin_num_pair)-1:
            #     j = (j+1)%len(size_pin_num_pair)
            #     k = (k+1)%len(member_to_copy) #选member复制（dates）
            # else:
            #     j = (j+1)%len(size_pin_num_pair) #选size pin num复制
            # print(f"j:{j}, loop_num:{loop_num}, len(studytype_user_date_size_pin_num_pair_to_copy):{len(studytype_user_date_size_pin_num_pair_to_copy)}, len(users_to_copy):{len(users_to_copy)}")
            # if j == len(studytype_user_date_size_pin_num_pair_to_copy)-1:
            #     i= (i+1)%len(users_to_copy)
            #     j = (j+1)%len(studytype_user_date_size_pin_num_pair_to_copy)
            # else:
            j = (j + 1) % len(studytype_user_date_size_pin_num_pair_to_copy)

            loop_num += 1

        # 生成高斯噪声并添加到选定的正类样本
        noise_scale = noise_level * positive_features_to_augment.std()  # 调整噪声水平
        gaussian_noise = np.random.normal(0, noise_scale, positive_features_to_augment.shape)
        positive_features_noisy = positive_features_to_augment + gaussian_noise

        # 将增强的样本合并回原始数据集
        result_array_augmented = np.concatenate((result_array, positive_features_noisy), axis=0)
        # label_augmented = np.concatenate((labels, labels[indices_to_copy]), axis=0)
        binary_labels_augmented = np.concatenate((binary_labels, [1 for _ in range(positive_samples_needed)]), axis=0)

        # 重新缩放数据 1231update 弃用
        # if scaler_augment is None:
        #     scaler_augment = StandardScaler()
        #     scaler_augment.fit(result_array_augmented)
        # scaled_data_augmented = scaler_augment.transform(result_array_augmented)
        scaled_data_augmented = result_array_augmented
    else:
        # 如果不需要增加正样本，则保持原始数据不变
        scaled_data_augmented = scaled_data
        # label_augmented = labels
        binary_labels_augmented = binary_labels

    # 返回原始和增强后的数据和标签,以及scaler_origin, scaler_augment
    # print(f"labels:{labels}")
    # print(f"binary_labels:{binary_labels}")
    # print(f"binary_labels_augmented:{binary_labels_augmented}")
    return scaled_data, np.array(labels), np.array(binary_labels), scaled_data_augmented, np.array(
        binary_labels_augmented), scaler_origin, scaler_augment


################################################################ knn 二分类
def knn4con_binary(n_neighbors=3,
                   data_scaled=None, binary_labels=None, latter_data_scaled=None, latter_labels=None):
    # 生成示例数据
    # labels = np.repeat(np.arange(num_people), authentications_per_person)

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(data_scaled, binary_labels, test_size=0.2)
    # X_test = np.concatenate((X_test, latter_data_scaled), axis=0)
    # y_test = np.concatenate((y_test, latter_labels), axis=0)
    # print("ytest", y_test)
    # print("testing shape:", X_test.shape)

    # 1231update 数据的标准化做在train set上
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    latter_data_scaled = scaler.transform(latter_data_scaled)

    # 创建KNN模型
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    # print("ypre", y_pred)

    # y_special_pred = knn_model.predict(latter_data_scaled)
    # print("y_special_pred", y_special_pred)

    # 准确度
    accuracy = accuracy_score(y_test, y_pred)
    print("准确度:", accuracy)

    # 混淆矩阵
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("混淆矩阵:")
    print(conf_matrix)

    # 精确度、召回率、F1分数
    # precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=True)
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

    # if not np.isnan(latter_data_scaled):
    latter_y_pred = knn_model.predict(latter_data_scaled)
    print(latter_y_pred, latter_labels)

    latter_accuracy = accuracy_score(latter_y_pred, latter_labels)
    print('随时间推移的准确率', latter_accuracy)


################################################################ knn 多分类
def knn4con_multi(n_neighbors=3,
                  data_scaled=None, labels=None, latter_data_scaled=None, latter_labels=None):
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(data_scaled, labels, test_size=0.2)
    # print("testing shape:", X_test.shape)

    # 1231update 数据的标准化做在train set上
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    latter_data_scaled = scaler.transform(latter_data_scaled)

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

    # if not np.isnan(latter_data_scaled):
    latter_y_pred = knn_model.predict(latter_data_scaled)
    print(latter_y_pred, latter_labels)

    latter_accuracy = accuracy_score(latter_y_pred, latter_labels)
    print('随时间推移的准确率', latter_accuracy)

    # # 分类报告
    # class_report = classification_report(y_test, y_pred)
    # print("分类报告:")
    # print(class_report)


################################################################ svm 二分类
def svm4con_binary(kernel="linear", C=1, gamma=0.02,
                   data_scaled=None, binary_labels=None
                   , latter_data_scaled=None, latter_labels=None):
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(data_scaled, binary_labels, test_size=0.2)
    # print("testing shape:", X_test.shape)

    # 1231update 数据的标准化做在train set上
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    latter_data_scaled = scaler.transform(latter_data_scaled)

    # 创建svm模型
    svm_model = SVC(kernel=kernel, C=C, gamma=gamma, )
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

    # if not np.isnan(latter_data_scaled):
    latter_y_pred = svm_model.predict(latter_data_scaled)
    print(latter_y_pred, latter_labels)

    latter_accuracy = accuracy_score(latter_y_pred, latter_labels)
    print('随时间推移的准确率', latter_accuracy)


################################################################ svm 多分类
def svm4con_multi(kernel="linear", C=1, gamma=0.02,
                  data_scaled=None, labels=None, latter_data_scaled=None, latter_labels=None):
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(data_scaled, labels, test_size=0.2)

    # 1231update 数据的标准化做在train set上
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    latter_data_scaled = scaler.transform(latter_data_scaled)

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
    # if not np.isnan(latter_data_scaled):
    latter_y_pred = svm_model.predict(latter_data_scaled)
    print(latter_y_pred, latter_labels)

    latter_accuracy = accuracy_score(latter_y_pred, latter_labels)
    print('随时间推移的准确率', latter_accuracy)


################################################################ knn kfold 二分类
def knn4con_binary_kfolds(n_neighbors=3, n_splits=5, data_scaled=None, binary_labels=None):
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

        # 1231update 数据的标准化做在train set上
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

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
def knn4con_multi_kfolds(n_neighbors=3, n_splits=5, data_scaled=None, labels=None):
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

        # 1231update 数据的标准化做在train set上
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

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
def svm4con_binary_kfolds(kernel="linear", C=1, gamma=0.02, n_splits=3, data_scaled=None, binary_labels=None):
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

        # 1231update 数据的标准化做在train set上
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

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
def svm4con_multi_kfolds(kernel="linear", C=1, gamma=0.02, n_splits=3, data_scaled=None, labels=None):
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

        # 1231update 数据的标准化做在train set上
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

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
    os.chdir('D:\pycharm\srt_vr_auth') # cwd的绝对路径
    positive_label = ['7']  # 正样本
    model = 'head'  # model
    n_split = 3  # k fold
    kernel = "linear" # svm

    data_scaled, labels, binary_labels, scaled_data_augmented, binary_labels_augmented, _, _ = data_scaled_and_label(
        default_authentications_per_person=9, rotdir=os.path.join(os.getcwd(), "data/"), positive_label=positive_label,
        model=model, studytype_users_dates_range=read_data_name_from_json()[0], size_list=[3], pin_list=[14],
        noise_level=0.0001)

    print(f"labels:{labels}")
    print(f"binary_labels:{binary_labels}")
    print(f"binary_labels_augmented:{binary_labels_augmented}")
    print(f"data_scaled:{data_scaled.shape}")
    # print(f"scaler_origin:{scaler_origin.mean_}, scaler_augment:{scaler_augment.mean_}")
    # 原数据和标签跑机器学习模型
    print("")
    print("original data")

    print("---------knn_binary_kfold------------")
    knn4con_binary_kfolds(data_scaled=data_scaled, binary_labels=binary_labels,
                          n_splits=n_split)

    print("---------knn_multi_kfold------------")
    knn4con_multi_kfolds(data_scaled=data_scaled, labels=labels,
                         n_splits=n_split)

    print("----------svm_binary_kfold------------")
    svm4con_binary_kfolds(data_scaled=data_scaled, binary_labels=binary_labels,
                          n_splits=n_split, kernel=kernel)

    print("-----------svm_multi_kfold------------")
    svm4con_multi_kfolds(data_scaled=data_scaled, labels=labels,
                         n_splits=n_split, kernel=kernel)

    # 数据增强后的数据和标签跑模型
    print("")
    print("augmented data")

    print("---------knn_binary_kfold------------")
    knn4con_binary_kfolds(data_scaled=scaled_data_augmented, binary_labels=binary_labels_augmented,
                          n_splits=n_split)

    print("----------svm_binary_kfold------------")
    svm4con_binary_kfolds(data_scaled=scaled_data_augmented, binary_labels=binary_labels_augmented,
                          n_splits=n_split, kernel=kernel)


    ################################ 随时间推移重新检验部分
    # 日后新采的数据属性
    default_latter_auth_per_person = 4  # 每人采集次数
    latter_positive_label = positive_label  # 正样本, 与之前是一致的

    latter_data_scaled, latter_labels, latter_binary_labels, _, _, _, _ = data_scaled_and_label(
        default_authentications_per_person=default_latter_auth_per_person, rotdir=os.path.join(os.getcwd(), "data/"),
        positive_label=latter_positive_label, model=model,
        studytype_users_dates_range=read_data_name_from_json()[1],
        size_list=[3], pin_list=[14], noise_level=0.0001, scaler_origin=None, scaler_augment=None)

    print("")
    print(f"latter_data_scaled: {latter_data_scaled.shape}")
    print("")

    print("--------knn_binary------------")
    knn4con_binary(data_scaled=scaled_data_augmented, binary_labels=binary_labels_augmented,
                   latter_data_scaled=latter_data_scaled, latter_labels=latter_binary_labels)

    print("---------knn_multi------------")
    knn4con_multi(data_scaled=data_scaled, labels=labels, latter_data_scaled=latter_data_scaled,
                  latter_labels=latter_labels)

    print("---------svm_binary------------")
    svm4con_binary(data_scaled=scaled_data_augmented, binary_labels=binary_labels_augmented,
                   latter_data_scaled=latter_data_scaled, latter_labels=latter_binary_labels)

    print("---------svm_multi------------")
    svm4con_multi(data_scaled=data_scaled, labels=labels,
                  latter_data_scaled=latter_data_scaled, latter_labels=latter_labels
                  )


if __name__ == "__main__":
    with open('output.txt', 'w') as file:  # 将print内容保存到文件
        # 保存当前的标准输出
        original_stdout = sys.stdout
        # 将标准输出重定向到文件
        sys.stdout = file
        main()
        # 恢复原来的标准输出
        sys.stdout = original_stdout

    # main() # 用于在终端输出print内容

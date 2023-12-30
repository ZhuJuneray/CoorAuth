import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, classification_report
import os, json, re
from data_preprocess import difference_gaze_head, add_noise, data_zero_smooth_feature, merged_array_generator
from sklearn.svm import SVC


def smooth_data(arr, window_parameter=9, polyorder_parameter=2):
    arr_smoothed = savgol_filter(arr, window_length=window_parameter, polyorder=polyorder_parameter)
    return arr_smoothed


def read_data_name_from_json(filepath = "src/data.json"):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    data_list = [f"{item['studytype']}-{item['names']}-{item['date']}" for item in data['data']]
    latter_auth_list = [f"{item['studytype']}-{item['names']}-{item['date']}" for item in data['latter_auth']]
    return data_list, latter_auth_list


def add_small_noise(sequence, noise_level=0.01):
    noise = np.random.normal(0, noise_level, len(sequence))
    augmented_sequence = sequence + noise
    return augmented_sequence


def data_scaled_and_label(studytype_users_dates, rotdir=None, model="", size_num_study1= [1,2,3,4,5,6],
                           size_num_study2 = [3],  pin_num = [1,2,3,4], authentications_per_person=6,
                            positive_label=None, noise_level =0.1): # 返回scaled后的原始数据和标签，scaled后的增强后的数据和标签
    import itertools
    def map_names_to_numbers(names): # 将名字映射为数字，按顺序从1开始，如果名字重复，数字也重复，如：['a','b','a'] -> [1,2,1]
        name_to_number = {}
        number_list = []
        counter = 1
        for name in names:
            if name not in name_to_number:
                name_to_number[name] = counter
                counter += 1
            number_list.append(name_to_number[name])
        return number_list
    
    size_pin_num_pair=[] # size, pin, num 的所有排列组合，用于数据增强时循环增强所有正标签里的数据
    result_array = np.array([])
    for member in studytype_users_dates:
        if member.split('-')[0] == 'study1': # 因为study1和study2的studytype_users_dates的内容不一样，所以分开处理
            size_pin_num_pair = [x for x in itertools.product(size_num_study1, pin_num,range(1,authentications_per_person+1))]
        # 特征:
            for size in size_num_study1:
                for pin in pin_num:
                    for num in range(authentications_per_person):
                        merged_array = merged_array_generator(member=member, rotdir=rotdir, model=model, size=size, pin=pin, num=num, noise_flag=False) # 返回该member, size, pin, num的特征, 返回的是可以直接用于机器学习的一维X向量
                        result_array = np.vstack([result_array, merged_array]) if result_array.size else merged_array # 将所有特征堆叠起来，每一行是一个特征

        if member.split('-')[0] == 'study2':
            size_pin_num_pair = [x for x in itertools.product(size_num_study2, pin_num,range(1,authentications_per_person+1))]
        # 特征:
            for size in size_num_study2:
                for pin in pin_num:
                    for num in range(authentications_per_person):
                        merged_array = merged_array_generator(member=member, rotdir=rotdir, model=model, size=size, pin=pin, num=num, noise_flag=False)
                        result_array = np.vstack([result_array, merged_array]) if result_array.size else merged_array

    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(result_array) # 标准化数据

    users = [x.split('-')[1] for x in studytype_users_dates]
    num_people = len(map_names_to_numbers(users))
    if studytype_users_dates[0].split('-')[0] == 'study1':
        labels = np.repeat(map_names_to_numbers(users), authentications_per_person*len(size_num_study1)*len(pin_num))
        binary_labels = np.array([1 if user in positive_label else 0 for user in users for _ in range(authentications_per_person*len(size_num_study1)*len(pin_num))])
    if studytype_users_dates[0].split('-')[0] == 'study2':
        labels = np.repeat(map_names_to_numbers(users), authentications_per_person*len(size_num_study2)*len(pin_num))
        binary_labels = np.array([1 if user in positive_label else 0 for user in users for _ in range(authentications_per_person*len(size_num_study2)*len(pin_num))])

    # 识别正类样本
    positive_indices = np.where(binary_labels == 1)[0]
    positive_features = result_array[positive_indices]

    # 确定增强的正样本数量以达到大约50%的正样本比例
    total_samples_needed = len(binary_labels)  # 总样本数
    positive_samples_needed = int(total_samples_needed) - 2*len(positive_indices)  # 需要增强的正样本数

    # 如果需要增强的样本数为负数或零，则不执行任何操作
    if positive_samples_needed > 0:
        # 选择正样本进行复制和添加噪声
        users_to_copy = np.random.choice(positive_label, size=positive_samples_needed, replace=True)
        loop_num = 0
        i=0
        j=0
        k=0
        positive_features_to_augment = np.array([])
        
        while loop_num < positive_samples_needed:
            user_to_copy = users_to_copy[i]
            pattern = '^'+studytype_users_dates[0].split('-')[0]+f'-{user_to_copy}-.*$'
            member_to_copy = [x for x in studytype_users_dates if re.match(pattern, x)]
            merged_array_augmented = merged_array_generator(member=member_to_copy[k], rotdir=rotdir, model=model, size=size_pin_num_pair[j][0], pin=size_pin_num_pair[j][1], num=size_pin_num_pair[j][2]-1, noise_flag=True, noise_level=noise_level)
            positive_features_to_augment = np.vstack([positive_features_to_augment, merged_array_augmented]) if positive_features_to_augment.size else merged_array_augmented
            if k == len(member_to_copy)-1 and j == len(size_pin_num_pair)-1:
                i= (i+1)%len(users_to_copy) #选user复制
                j = (j+1)%len(size_pin_num_pair)
                k = (k+1)%len(member_to_copy)
            elif j == len(size_pin_num_pair)-1:
                j = (j+1)%len(size_pin_num_pair)
                k = (k+1)%len(member_to_copy) #选member复制（dates）
            else:
                j = (j+1)%len(size_pin_num_pair) #选size pin num复制
            loop_num+=1

        # 生成高斯噪声并添加到选定的正类样本
        noise_scale = noise_level * positive_features_to_augment.std()  # 调整噪声水平
        gaussian_noise = np.random.normal(0, noise_scale, positive_features_to_augment.shape)
        positive_features_noisy = positive_features_to_augment + gaussian_noise

        # 将增强的样本合并回原始数据集
        result_array_augmented = np.concatenate((result_array, positive_features_noisy), axis=0)
        # label_augmented = np.concatenate((labels, labels[indices_to_copy]), axis=0)
        binary_labels_augmented = np.concatenate((binary_labels, np.ones(positive_samples_needed)), axis=0)

        # 重新缩放数据
        scaler = StandardScaler()
        scaled_data_augmented = scaler.fit_transform(result_array_augmented)
    else:
        # 如果不需要增加正样本，则保持原始数据不变
        scaled_data_augmented = scaled_data
        # label_augmented = labels
        binary_labels_augmented = binary_labels

    # 返回原始和增强后的数据和标签
    return scaled_data, labels, binary_labels, scaled_data_augmented,binary_labels_augmented


################################################################ knn 二分类
def knn4con_binary( model, n_neighbors=3,
                    data_scaled=None, binary_labels=None, latter_data_scaled=None, latter_labels=None):
    # 生成示例数据
    # labels = np.repeat(np.arange(num_people), authentications_per_person)

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(data_scaled, binary_labels, test_size=0.2)
    X_test = np.concatenate((X_test, latter_data_scaled), axis=0)
    y_test = np.concatenate((y_test, latter_labels), axis=0)
    print("ytest", y_test)
    # print("testing shape:", X_test.shape)

    # 创建KNN模型
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    print("ypre", y_pred)

    y_special_pred = knn_model.predict(latter_data_scaled)
    print("y_special_pred", y_special_pred)


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
def knn4con_multi(model, n_neighbors=3,
                    data_scaled=None, labels=None, latter_data_scaled=None, latter_labels=None):
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
def svm4con_binary(model, kernel="linear", C=1, gamma=0.02,
                    data_scaled=None, binary_labels=None
                   ,latter_data_scaled=None, latter_labels=None):
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


    # if not np.isnan(latter_data_scaled):
    latter_y_pred = svm_model.predict(latter_data_scaled)
    print(latter_y_pred, latter_labels)

    latter_accuracy = accuracy_score(latter_y_pred, latter_labels)
    print('随时间推移的准确率', latter_accuracy)


################################################################ svm 多分类
def svm4con_multi(model, kernel="linear", C=1, gamma=0.02,
                 data_scaled=None, labels=None, latter_data_scaled=None, latter_labels=None):
    
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
    # if not np.isnan(latter_data_scaled):
    latter_y_pred = svm_model.predict(latter_data_scaled)
    print(latter_y_pred, latter_labels)

    latter_accuracy = accuracy_score(latter_y_pred, latter_labels)
    print('随时间推移的准确率', latter_accuracy)


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

    positive_label = ['7']  # 正样本
    data_scaled, labels, binary_labels, scaled_data_augmented, binary_labels_augmented  = data_scaled_and_label(authentications_per_person=9, rotdir = os.path.join(os.getcwd(), "data/"), positive_label=positive_label, model="head+eye", studytype_users_dates=read_data_name_from_json()[0], size_num_study2=[3], pin_num=range(13,19), noise_level=0.0001)
    model = ''  # model
    n_split = 2  # k fold

    # print(f"labels:{labels}")
    # print(f"binary_labels:{binary_labels}")
    # print(f"binary_labels_augmented:{binary_labels_augmented}")
    
    kernel = "linear"
    print(data_scaled.shape)

    # 原数据和标签跑机器学习模型
    print("")
    print("original data")

    print("---------knn_binary_kfold------------")
    knn4con_binary_kfolds(data_scaled=data_scaled, binary_labels=binary_labels,
                          model=model, n_splits=n_split)

    print("---------knn_multi_kfold------------")
    knn4con_multi_kfolds(data_scaled=data_scaled, labels=labels,
                         model=model,
                         n_splits=n_split)

    print("----------svm_binary_kfold------------")
    svm4con_binary_kfolds(data_scaled=data_scaled, binary_labels=binary_labels,
                           n_splits=n_split, kernel=kernel)

    print("-----------svm_multi_kfold------------")
    svm4con_multi_kfolds(data_scaled=data_scaled, labels=labels,
                         model=model,
                         n_splits=n_split, kernel=kernel)
    
    # 数据增强后的数据和标签跑模型
    print("")
    print("augmented data")

    print("---------knn_binary_kfold------------")
    knn4con_binary_kfolds(data_scaled=scaled_data_augmented, binary_labels=binary_labels_augmented,
                          model=model, n_splits=n_split)
    
    print("----------svm_binary_kfold------------")
    svm4con_binary_kfolds(data_scaled=scaled_data_augmented, binary_labels=binary_labels_augmented,
                           n_splits=n_split, kernel=kernel)
    

    ################################ 随时间推移重新检验部分

    # 日后新采的数据属性
    latter_auth_per_person = 4 # 每人采集次数
    latter_positive_label = ['7']  # 正样本


    latter_data_scaled, latter_labels, latter_binary_labels, _,_  = data_scaled_and_label(authentications_per_person=latter_auth_per_person, rotdir = os.path.join(os.getcwd(), "data/"), positive_label=latter_positive_label, model="head+eye", studytype_users_dates=read_data_name_from_json()[1], size_num_study2=[3], pin_num=range(13,19), noise_level=0.0001)
    

    print(f"latter_data_scaled: {latter_data_scaled.shape}")

    print("--------knn_binary------------")
    knn4con_binary(data_scaled=scaled_data_augmented, binary_labels=binary_labels_augmented,
                   model=model, latter_data_scaled=latter_data_scaled, latter_labels=latter_binary_labels)

    print("---------knn_multi------------")
    knn4con_multi(model=model, data_scaled=data_scaled, labels=labels, latter_data_scaled=latter_data_scaled, latter_labels=latter_labels)

    print("---------svm_binary------------")
    svm4con_binary( data_scaled=scaled_data_augmented, binary_labels=binary_labels_augmented,
                   model=model, latter_data_scaled=latter_data_scaled, latter_labels=latter_binary_labels)

    print("---------svm_multi------------")
    svm4con_multi(data_scaled=data_scaled, labels=labels,
                   model=model, latter_data_scaled=latter_data_scaled, latter_labels=latter_labels
                  )



if __name__ == "__main__":
    main()
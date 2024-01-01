import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, classification_report
import os, json, re, sys
from data_preprocess import data_augment_and_label, read_data_latter_data_json
from sklearn.svm import SVC


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
    # os.chdir('D:\pycharm\srt_vr_auth')  # cwd的绝对路径
    positive_label = ['7']  # 正样本
    model = 'head'  # model
    n_split = 3  # k fold
    kernel = "linear"  # svm
    noise_level = 0.3  # noise level
    augmentation_time = 10  # 高斯噪声做数据增强的倍数

    # 1.1update augment_time表示增强为原来数量的多少倍，如果留空则为默认值1，即全部为原始数据
    data_scaled, labels, binary_labels, scaled_data_augmented, binary_labels_augmented = data_augment_and_label(
        default_authentications_per_person=9, rotdir=os.path.join(os.getcwd(), "data/"), positive_label=positive_label,
        model=model, studytype_users_dates_range=read_data_latter_data_json()[0], size_list=[3], pin_list=[14],
        noise_level=noise_level, augment_time=augmentation_time)

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

    latter_data_scaled, latter_labels, latter_binary_labels, _, _ = data_augment_and_label(
        default_authentications_per_person=default_latter_auth_per_person, rotdir=os.path.join(os.getcwd(), "data/"),
        positive_label=latter_positive_label, model=model,
        studytype_users_dates_range=read_data_latter_data_json()[1],
        size_list=[3], pin_list=[14], noise_level=noise_level)

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

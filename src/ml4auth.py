from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, classification_report
from selection_calculate import calculate_frr_far


################################################################ knn 二分类
def knn_binary(n_neighbors=3, data_scaled=None, binary_labels=None, latter_data_scaled=None, latter_labels=None,
               test_size=0.2):
    # 生成示例数据
    # labels = np.repeat(np.arange(num_people), authentications_per_person)

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(data_scaled, binary_labels, test_size=test_size)
    # X_test = np.concatenate((X_test, latter_data_scaled), axis=0)
    # y_test = np.concatenate((y_test, latter_labels), axis=0)
    # print("ytest", y_test)
    print("shape:", X_train.shape, X_test.shape)

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
def knn_multi(n_neighbors=3, data_scaled=None, labels=None, latter_data_scaled=None, latter_labels=None,
              test_size=0.2):
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(data_scaled, labels, test_size=test_size, stratify=labels)
    # print("testing shape:", X_test.shape)

    # 获取排序后的索引
    train_sort_indices = np.argsort(y_train)
    test_sort_indices = np.argsort(y_test)

    # 使用索引重新排序 X_train 和 X_test
    X_train = X_train[train_sort_indices]
    X_test = X_test[test_sort_indices]

    # 使用排序后的 y_train 和 y_test
    y_train = np.sort(y_train)
    y_test = np.sort(y_test)

    # 1231update 数据的标准化做在train set上
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 创建KNN模型
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)

    # 准确度
    accuracy = accuracy_score(y_test, y_pred)
    print("accuracy:", accuracy)

    # 混淆矩阵
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("confusion:")
    print(conf_matrix)

    # 精确度、召回率、F1分数
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    # precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    print("end")
    print("precision:", precision)
    print("recall:", recall)
    print("f1:", f1)

    if latter_labels is not None:
        latter_data_scaled = scaler.transform(latter_data_scaled)
        # if not np.isnan(latter_data_scaled):
        latter_y_pred = knn_model.predict(latter_data_scaled)
        print(latter_y_pred, latter_labels)

        latter_accuracy = accuracy_score(latter_y_pred, latter_labels)
        print('随时间推移的准确率', latter_accuracy)


################################################################ svm 二分类
def svm_binary(kernel="linear", C=1, gamma=0.02, data_scaled=None, binary_labels=None,
               latter_data_scaled=None, latter_labels=None, test_size=0.2):
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(data_scaled, binary_labels, test_size=test_size)
    # print("testing shape:", X_test.shape)

    # 1231update 数据的标准化做在train set上
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    latter_data_scaled = scaler.transform(latter_data_scaled)

    # 创建svm模型
    svm_model = SVC(kernel=kernel, C=C, gamma=gamma, probability=True)
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
    latter_confidence = svm_model.decision_function(latter_data_scaled)
    print(latter_y_pred, latter_labels)
    print("Confidence", latter_confidence)

    latter_accuracy = accuracy_score(latter_y_pred, latter_labels)
    print('随时间推移的准确率', latter_accuracy)


################################################################ svm 多分类
def svm_multi(kernel="linear", C=1, gamma=0.02, data_scaled=None, labels=None,
              latter_data_scaled=None, latter_labels=None, test_size=0.2):
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(data_scaled, labels, test_size=test_size, stratify=labels)

    # # 初始化空的测试集和训练集
    # X_test = []
    # y_test = []
    # X_train = []
    # y_train = []
    # # 设定测试集和训练集的大小比例
    # test_size = 20
    # train_size = 20
    #
    # # 计算总的测试集和训练集的个数
    # num_samples = len(data_scaled)
    # print('num_samples', num_samples)
    # num_test_sets = num_samples // (test_size + train_size)
    #
    # # 数据标准化
    # scaler = StandardScaler()
    #
    # # 循环划分数据集
    # for i in range(num_test_sets):
    #     start_idx = i * (test_size + train_size)
    #     end_idx_test = start_idx + test_size
    #     end_idx_train = end_idx_test + train_size
    #
    #     # 划分测试集
    #     X_test_set = data_scaled[start_idx:end_idx_test]
    #     y_test_set = labels[start_idx:end_idx_test]
    #
    #     # 划分训练集
    #     X_train_set = data_scaled[end_idx_test:end_idx_train]
    #     y_train_set = labels[end_idx_test:end_idx_train]
    #     # print("X_train_start", end_idx_test, "X_train_end", end_idx_train)
    #     # 将划分好的测试集和训练集添加到总集合中
    #     print("labels", y_test_set, " ", y_train_set)
    #     X_test.append(X_test_set)
    #     y_test.append(y_test_set)
    #     X_train.append(X_train_set)
    #     y_train.append(y_train_set)
    #
    # # 将列表转换为 NumPy 数组
    # X_test = np.concatenate(X_test, axis=0)
    # y_test = np.concatenate(y_test, axis=0)
    # X_train = np.concatenate(X_train, axis=0)
    # y_train = np.concatenate(y_train, axis=0)
    #
    # X_train, y_train = shuffle(X_train, y_train)
    # X_test, y_test = shuffle(X_test, y_test)

    # train_sort_indices = np.argsort(y_train)
    # test_sort_indices = np.argsort(y_test)
    #
    # # 使用索引重新排序 X_train 和 X_test
    # X_train = X_train[train_sort_indices]
    # X_test = X_test[test_sort_indices]
    #
    # # 使用排序后的 y_train 和 y_test
    # y_train = np.sort(y_train)
    # y_test = np.sort(y_test)

    # 1231update 数据的标准化做在train set上
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    # 创建svm模型
    svm_model = SVC(kernel=kernel, C=C, gamma=gamma, probability=True)
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)

    # 准确度
    accuracy = accuracy_score(y_test, y_pred)
    print("accuracy:", accuracy)

    # 混淆矩阵
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("confusion:")
    print(conf_matrix)
    frr_list = []
    far_list = []
    for i in range(17):
        frr, far = calculate_frr_far(conf_matrix, i)
        frr_list.append(frr)
        far_list.append(far)

    frr_mean = np.mean(frr_list)
    far_mean = np.mean(far_list)
    frr_std = np.std(frr_list) / np.sqrt(len(frr_list))
    far_std = np.std(far_list) / np.sqrt(len(far_list))
    print(f"Frr: {frr_mean}, Far: {far_mean}, StdFrr: {frr_std}, StdFar: {far_std}")

    # 精确度、召回率、F1分数
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    # precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    print("end")
    print("precision:", precision)
    print("recall:", recall)
    print("f1:", f1)

    if latter_labels is not None:
        latter_data_scaled = scaler.transform(latter_data_scaled)
        # if not np.isnan(latter_data_scaled):
        latter_y_pred = svm_model.predict(latter_data_scaled)
        latter_confidence = svm_model.decision_function(latter_data_scaled)
        print(latter_y_pred, latter_labels)
        print("Confidence", latter_confidence)

        latter_accuracy = accuracy_score(latter_y_pred, latter_labels)
        print('随时间推移的准确率', latter_accuracy)


############################################################### rf
def rf_multi(n_estimators=100, data_scaled=None, labels=None,
              latter_data_scaled=None, latter_labels=None, test_size=0.2):
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(data_scaled, labels, test_size=test_size, stratify=labels)
    # 1231update 数据的标准化做在train set上
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    # 创建rf模型
    rf_model = RandomForestClassifier(n_estimators=n_estimators)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)

    # 准确度
    accuracy = accuracy_score(y_test, y_pred)
    print("accuracy:", accuracy)

    # 混淆矩阵
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("confusion:")
    print(conf_matrix)
    frr_list = []
    far_list = []
    for i in range(17):
        frr, far = calculate_frr_far(conf_matrix, i)
        frr_list.append(frr)
        far_list.append(far)

    frr_mean = np.mean(frr_list)
    far_mean = np.mean(far_list)
    frr_std = np.std(frr_list) / np.sqrt(len(frr_list))
    far_std = np.std(far_list) / np.sqrt(len(far_list))
    print(f"Frr: {frr_mean}, Far: {far_mean}, StdFrr: {frr_std}, StdFar: {far_std}")

    # 精确度、召回率、F1分数
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    # precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    print("end")
    print("precision:", precision)
    print("recall:", recall)
    print("f1:", f1)

    if latter_labels is not None:
        latter_data_scaled = scaler.transform(latter_data_scaled)
        # if not np.isnan(latter_data_scaled):
        latter_y_pred = rf_model.predict(latter_data_scaled)
        # latter_confidence = rf_model.decision_function(latter_data_scaled) #  for the random forest model, it doesn't have the confidence
        print(latter_y_pred, latter_labels)
        # print("Confidence", latter_confidence)

        latter_accuracy = accuracy_score(latter_y_pred, latter_labels)
        print('随时间推移的准确率', latter_accuracy)


################################################################ knn kfold 二分类
def knn_binary_kfolds(n_neighbors=3, n_splits=5, data_scaled=None, binary_labels=None):
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
    # print("Average f1s", average_f1s)


################################################################ knn kfold 多分类
def knn_multi_kfolds(n_neighbors=3, n_splits=5, data_scaled=None, labels=None):
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
        print(X_train.shape, X_test.shape)
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
    print("Average f1s", average_f1s)


################################################################ svm kfold 二分类
def svm_binary_kfolds(kernel="linear", C=1, gamma=0.02, n_splits=3, data_scaled=None, binary_labels=None):
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
    # print("Average f1s", average_f1s)


################################################################ svm kfold 多分类
def svm_multi_kfolds(kernel="linear", C=1, gamma=0.02, n_splits=3, data_scaled=None, labels=None):
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


################################################################ rf kfold 多分类
def rf_multi_kfolds(n_estimators=100, n_splits=3, data_scaled=None, labels=None, latter_data_scaled=None, latter_labels=None):
    for label in np.unique(labels):
        count = np.sum(labels == label)
        # print(f"Class {label}: {count} samples")
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True)

    # 初始化准确性列表，用于存储每个折叠的模型准确性
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    fars = []
    frrs = []
    
    accuracies_latter = []
    precisions_latter = []
    recalls_latter = []
    f1s_latter = []
    fars_latter = []
    frrs_latter = []

    for train_index, test_index in kf.split(data_scaled, labels):
        X_train, X_test = data_scaled[train_index], data_scaled[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        # 1231update 数据的标准化做在train set上
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # 创建模型
        rf_model = RandomForestClassifier(n_estimators=n_estimators)

        # 模型训练
        rf_model.fit(X_train, y_train)

        # 模型预测
        y_pred = rf_model.predict(X_test)

        # 计算准确性
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=1)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        # conf_matrix = confusion_matrix(y_test, y_pred)
        # true_positive = conf_matrix[1, 1]
        # false_positive = conf_matrix[0, 1]
        # true_negative = conf_matrix[0, 0]
        # false_negative = conf_matrix[1, 0]

        # # 计算 FAR 和 FRR
        # far = false_positive / (false_positive + true_negative)
        # frr = false_negative / (false_negative + true_positive)

        # fars.append(far)
        # frrs.append(frr)

        # 随时间推移的准确率
        if latter_data_scaled is not None:
            latter_y_pred = rf_model.predict(latter_data_scaled)
            latter_accuracy = accuracy_score(latter_y_pred, latter_labels)
            accuracies_latter.append(latter_accuracy)
            latter_precision, latter_recall, latter_f1, _ = precision_recall_fscore_support(latter_labels, latter_y_pred, average='weighted', zero_division=1)
            precisions_latter.append(latter_precision)
            recalls_latter.append(latter_recall)
            f1s_latter.append(latter_f1)

            print("latter_y_pred, latter_labels")
            print(latter_y_pred, latter_labels)

            # conf_matrix_latter = confusion_matrix(latter_labels, latter_y_pred)
            # true_positive_latter = conf_matrix_latter[1, 1]
            # false_positive_latter = conf_matrix_latter[0, 1]
            # true_negative_latter = conf_matrix_latter[0, 0]
            # false_negative_latter = conf_matrix_latter[1, 0]

            # # 计算 FAR 和 FRR
            # far_latter = false_positive_latter / (false_positive_latter + true_negative_latter)
            # frr_latter = false_negative_latter / (false_negative_latter + true_positive_latter)

            # fars_latter.append(far_latter)
            # frrs_latter.append(frr_latter)


    average_accuracy = np.mean(accuracies)
    average_precision = np.mean(precisions)
    average_recalls = np.mean(recalls)
    average_f1s = np.mean(f1s)
    average_fars = np.mean(fars)
    average_frrs = np.mean(frrs)

    print("Average Accuracy:", average_accuracy, "\nprecision:", average_precision, "\nrecalls:", average_recalls,
          "\nf1s:", average_f1s, "\nfars:", average_fars, "\nfrrs:", average_frrs)

    if latter_data_scaled is not None:
        average_accuracy_latter = np.mean(accuracies_latter)
        average_precision_latter = np.mean(precisions_latter)
        average_recalls_latter = np.mean(recalls_latter)
        average_f1s_latter = np.mean(f1s_latter)
        average_fars_latter = np.mean(fars_latter)
        average_frrs_latter = np.mean(frrs_latter)
        

        print("随时间推移的准确率")
        print("Average Accuracy Latter:", average_accuracy_latter, "\nprecision:", average_precision_latter, "\nrecalls:", average_recalls_latter,
              "\nf1s:", average_f1s_latter, "\nfars:", average_fars_latter, "\nfrrs:", average_frrs_latter)
    

################################################################ rf 二分类
def rf_binary_kfolds(n_estimators=100, n_splits=3, data_scaled=None, labels=None, latter_data_scaled=None, latter_labels=None):
    # 打印各类别的样本数量
    for label in np.unique(labels):
        count = np.sum(labels == label)
        print(f"Class {label}: {count} samples")

    # 设置交叉验证的折叠数
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True)

    # 初始化准确性列表，用于存储每个折叠的模型准确性
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    fars = []
    frrs = []
    
    accuracies_latter = []
    precisions_latter = []
    recalls_latter = []
    f1s_latter = []
    fars_latter = []
    frrs_latter = []

    # 进行交叉验证
    for train_index, test_index in kf.split(data_scaled, labels):
        X_train, X_test = data_scaled[train_index], data_scaled[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        # 数据的标准化做在train set上
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # 创建随机森林模型
        rf_model = RandomForestClassifier(n_estimators=n_estimators)

        # 模型训练
        rf_model.fit(X_train, y_train)

        # 模型预测
        y_pred = rf_model.predict(X_test)

        # 计算准确性
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=1)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        # conf_matrix = confusion_matrix(y_test, y_pred)
        # true_positive = conf_matrix[1, 1]
        # false_positive = conf_matrix[0, 1]
        # true_negative = conf_matrix[0, 0]
        # false_negative = conf_matrix[1, 0]

        # # 计算 FAR 和 FRR
        # far = false_positive / (false_positive + true_negative)
        # frr = false_negative / (false_negative + true_positive)

        # fars.append(far)
        # frrs.append(frr)

        # 随时间推移的准确率
        if latter_data_scaled is not None:
            latter_y_pred = rf_model.predict(latter_data_scaled)
            latter_accuracy = accuracy_score(latter_y_pred, latter_labels)
            accuracies_latter.append(latter_accuracy)
            latter_precision, latter_recall, latter_f1, _ = precision_recall_fscore_support(latter_labels, latter_y_pred, average='weighted', zero_division=1)
            precisions_latter.append(latter_precision)
            recalls_latter.append(latter_recall)
            f1s_latter.append(latter_f1)

            print("latter_y_pred, latter_labels")
            print(latter_y_pred, latter_labels)

            # conf_matrix_latter = confusion_matrix(latter_labels, latter_y_pred)
            # true_positive_latter = conf_matrix_latter[1, 1]
            # false_positive_latter = conf_matrix_latter[0, 1]
            # true_negative_latter = conf_matrix_latter[0, 0]
            # false_negative_latter = conf_matrix_latter[1, 0]

            # # 计算 FAR 和 FRR
            # far_latter = false_positive_latter / (false_positive_latter + true_negative_latter)
            # frr_latter = false_negative_latter / (false_negative_latter + true_positive_latter)

            # fars_latter.append(far_latter)
            # frrs_latter.append(frr_latter)

    # 计算平均指标
    average_accuracy = np.mean(accuracies)
    average_precision = np.mean(precisions)
    average_recalls = np.mean(recalls)
    average_f1s = np.mean(f1s)
    average_fars = np.mean(fars)
    average_frrs = np.mean(frrs)

    print("Average Accuracy:", average_accuracy, "\nPrecision:", average_precision, "\nRecalls:", average_recalls, "\nF1 Scores:", average_f1s, "\nFARs:", average_fars, "\nFRRs:", average_frrs)

    if latter_data_scaled is not None:
        average_accuracy_latter = np.mean(accuracies_latter)
        average_precision_latter = np.mean(precisions_latter)
        average_recalls_latter = np.mean(recalls_latter)
        average_f1s_latter = np.mean(f1s_latter)
        average_fars_latter = np.mean(fars_latter)
        average_frrs_latter = np.mean(frrs_latter)

        print("随时间推移的准确率")
        print("Average Accuracy Latter:", average_accuracy_latter, "\nPrecision:", average_precision_latter, "\nRecalls:", average_recalls_latter, "\nF1 Scores:", average_f1s_latter, "\nFARs:", average_fars_latter, "\nFRRs:", average_frrs_latter)
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, classification_report
import os, json, re, sys
from data_preprocess import data_augment_and_label, read_data_latter_data_json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


################################################################ mlp binary
def mlp_binary(hidden_layer_sizes=(256, ), activation='relu', alpha=0.0001, max_iter=200,
              data_scaled=None, binary_labels=None, latter_data_scaled=None, latter_labels=None, test_size=0.2):
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(data_scaled, binary_labels, test_size=test_size)

    # 数据标准化（仅在训练集上进行标准化，用相同的标准化器对测试集和未来数据进行标准化）
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    latter_data_scaled = scaler.transform(latter_data_scaled)

    # 创建MLP模型
    mlp_model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, alpha=alpha, max_iter=max_iter)
    mlp_model.fit(X_train, y_train)
    y_pred = mlp_model.predict(X_test)

    # 准确度
    accuracy = accuracy_score(y_test, y_pred)
    print("准确度:", accuracy)

    # 混淆矩阵
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("混淆矩阵:")
    print(conf_matrix)

    # 精确度、召回率、F1分数
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

    # 对未来数据进行预测
    latter_y_pred = mlp_model.predict(latter_data_scaled)
    print(latter_y_pred, latter_labels)

    latter_accuracy = accuracy_score(latter_y_pred, latter_labels)
    print('随时间推移的准确率', latter_accuracy)


################################################################ mlp multi
def mlp_multi(hidden_layer_sizes=(256, ), activation='relu', alpha=0.0001, max_iter=200,
              data_scaled=None, labels=None, latter_data_scaled=None, latter_labels=None, test_size=0.2):
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(data_scaled, labels, test_size=test_size, stratify=labels)

    # 数据标准化（仅在训练集上进行标准化，用相同的标准化器对测试集和未来数据进行标准化）
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    # 创建MLP模型
    mlp_model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, alpha=alpha, max_iter=max_iter)
    mlp_model.fit(X_train, y_train)
    y_pred = mlp_model.predict(X_test)

    # 准确度
    accuracy = accuracy_score(y_test, y_pred)
    print("accuracy:", accuracy)

    # 混淆矩阵
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("confusion:")
    print(conf_matrix)

    # 精确度、召回率、F1分数
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    print("end")
    print("precision:", precision)
    print("recall:", recall)
    print("f1:", f1)

    if latter_labels is not None:
        latter_data_scaled = scaler.transform(latter_data_scaled)
        # 对未来数据进行预测
        latter_y_pred = mlp_model.predict(latter_data_scaled)
        print(latter_y_pred, latter_labels)

        latter_accuracy = accuracy_score(latter_y_pred, latter_labels)
        print('随时间推移的准确率', latter_accuracy)


################################################################ mlp binary kfold
def mlp_binary_kfolds(hidden_layer_sizes=(256, ), activation='relu', solver='adam',
                      alpha=0.0001, max_iter=200, n_splits=5, data_scaled=None, binary_labels=None):
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

        mlp_model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation,
                                  solver=solver, alpha=alpha, max_iter=max_iter)
        mlp_model.fit(X_train, y_train)
        y_pred = mlp_model.predict(X_test)

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

    # 打印平均准确性 精确 召回 f1
    average_accuracy = np.mean(accuracies)
    average_precision = np.mean(precisions)
    average_recalls = np.mean(recalls)
    average_f1s = np.mean(f1s)
    average_fars = np.mean(fars)
    average_frrs = np.mean(frrs)
    print("平均准确性:", average_accuracy, "\n精确度:", average_precision, "\n召回率:", average_recalls,
          "\nF1 值:", average_f1s, "\nFAR:", average_fars, "\nFRR:", average_frrs)


################################################################ mlp multi kfold
def mlp_multi_kfolds(hidden_layer_sizes=(256, ), activation='relu', solver='adam', alpha=0.0001, max_iter=200,
                     n_splits=5, data_scaled=None, labels=None):
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

        mlp_model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation,
                                  solver=solver, alpha=alpha, max_iter=max_iter)
        mlp_model.fit(X_train, y_train)
        y_pred = mlp_model.predict(X_test)

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
    

################################################################ lstm binary
def lstm_binary(hidden_units=50, activation='tanh', dropout=0.2, recurrent_dropout=0.2, optimizer='adam', epochs=10, batch_size=4, 
                data_scaled=None, binary_labels=None, latter_data_scaled=None, latter_labels=None, test_size=0.2):
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(np.array(data_scaled), np.array(binary_labels), test_size=test_size)
    print(f"X_train.shape = ", X_train.shape)

    # 数据标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    if latter_data_scaled is not None:
        latter_data_scaled = scaler.transform(latter_data_scaled.reshape(-1, latter_data_scaled.shape[-1])).reshape(latter_data_scaled.shape)

    # 构建 LSTM 模型
    model = Sequential()
    model.add(LSTM(hidden_units, activation=activation, dropout=dropout, recurrent_dropout=recurrent_dropout, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1, activation='sigmoid'))  # 适用于二分类

    # 编译模型
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # 训练模型
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    # 预测
    y_pred = (model.predict(X_test) > 0.5).astype(int)

    # 准确度
    accuracy = accuracy_score(y_test, y_pred)
    print("准确度:", accuracy)

    # 混淆矩阵
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("混淆矩阵:")
    print(conf_matrix)

    # 精确度、召回率、F1分数
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=True)
    print("精确度:", precision)
    print("召回率:", recall)
    print("F1分数:", f1)

    # 从混淆矩阵中提取指标
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

    # 对未来数据进行预测
    latter_y_pred = (model.predict(latter_data_scaled) > 0.5).astype(int)
    print(latter_y_pred, latter_labels)

    latter_accuracy = accuracy_score(latter_labels, latter_y_pred)
    print('随时间推移的准确率:', latter_accuracy)

def lstm_binary_kfolds(hidden_units=50, activation='tanh', dropout=0.2, recurrent_dropout=0.2, optimizer='SGD', 
                       epochs=10, batch_size=4, n_splits=5, data_scaled=None, binary_labels=None):
    # 设置交叉验证的折叠数
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    
    # 初始化指标列表
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    fars = []
    frrs = []
    
    for train_index, test_index in kf.split(data_scaled, binary_labels):
        X_train, X_test = data_scaled[train_index], data_scaled[test_index]
        y_train, y_test = binary_labels[train_index], binary_labels[test_index]
        
        # 数据标准化
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        
        # 构建 LSTM 模型
        model = Sequential()
        model.add(LSTM(hidden_units, activation=activation, dropout=dropout, recurrent_dropout=recurrent_dropout, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dense(1, activation='sigmoid'))  # 适用于二分类
        
        # 编译模型
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        
        # 训练模型
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        
        # 预测
        y_pred = (model.predict(X_test) > 0.5).astype(int)
        
        # 计算指标
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
    
    # 打印平均指标
    print(f"lstm_binary_kfolds")
    print("平均准确性:", np.mean(accuracies))
    print("平均精确度:", np.mean(precisions))
    print("平均召回率:", np.mean(recalls))
    print("平均F1分数:", np.mean(f1s))
    print("平均FAR:", np.mean(fars))
    print("平均FRR:", np.mean(frrs))

def lstm_multi_kfolds(hidden_units=50, activation='tanh', dropout=0.2, recurrent_dropout=0.2, optimizer='SGD', 
                      epochs=10, batch_size=4, n_splits=5, data_scaled=None, labels=None):
    # 设置交叉验证的折叠数
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    
    # 初始化指标列表
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    
    for train_index, test_index in kf.split(data_scaled, labels):
        X_train, X_test = data_scaled[train_index], data_scaled[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        
        # 数据标准化
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        
        # One-hot 编码
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        
        # 构建 LSTM 模型
        model = Sequential()
        model.add(LSTM(hidden_units, activation=activation, dropout=dropout, recurrent_dropout=recurrent_dropout, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dense(y_train.shape[1], activation='softmax'))  # 适用于多分类
        
        # 编译模型
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        
        # 训练模型
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        
        # 预测
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)
        
        # 计算指标
        accuracy = accuracy_score(y_test_classes, y_pred_classes)
        accuracies.append(accuracy)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test_classes, y_pred_classes, average='weighted', zero_division=1)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    
    # 打印平均指标
    print(f"lstm_multi_kfolds")
    print("平均准确性:", np.mean(accuracies))
    print("平均精确度:", np.mean(precisions))
    print("平均召回率:", np.mean(recalls))
    print("平均F1分数:", np.mean(f1s))

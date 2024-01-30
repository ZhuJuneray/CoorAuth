# vote4auth.py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from selection_calculate import calculate_frr_far

def split_time_series(X, n):
    num_samples, num_features = X.shape
    num_segments = n

    # 创建一个空数组，用于存储分割后的时间序列
    X_split = np.zeros((num_samples, num_segments, num_features // num_segments))

    # 对每个时间序列进行分割
    for sample_index in range(num_samples):
        time_series = X[sample_index, :]

        # 根据余数分割
        for segment in range(num_segments):
            segment_indices = np.arange(segment, num_features, num_segments)
            X_split[sample_index, segment, :] = time_series[segment_indices]

    return X_split


def vote4auth(classifier=RandomForestClassifier(n_estimators=100), data_scaled=None, labels=None, test_size=0.2, n_segments=4):
    X_split = split_time_series(data_scaled, n_segments)

    X_train, X_test, y_train, y_test = train_test_split(X_split, labels, test_size=test_size, stratify=labels)

    print("X_shape", X_train.shape, X_test.shape)

    # 对每个段独立训练分类器
    classifiers = []
    scalers = []
    for segment in range(n_segments):
        # 提取训练数据的当前段
        X_train_segment = X_train[:, segment, :]
        # 训练分类器
        # classifier = KNeighborsClassifier(n_neighbors=3)
        # scaler = StandardScaler()
        # X_train_segment = scaler.fit_transform(X_train_segment)
        # scalers.append(scaler)
        classifier.fit(X_train_segment, y_train)
        classifiers.append(classifier)

    # for segment in range(n_segments):
    #     X_test[:, segment, :] = scalers[segment].transform(X_test[:, segment, :])

    # 对测试集进行投票
    final_predictions = []
    for sample in X_test:
        segment_predictions = [classifier.predict(sample[segment, :].reshape(1, -1))[0] for segment, classifier
                               in enumerate(classifiers)]
        print("segment", segment_predictions)
        final_prediction = max(set(segment_predictions), key=segment_predictions.count)  # 硬投票
        final_predictions.append(final_prediction)

    # 评估整体性能
    conf_matrix = confusion_matrix(y_test, final_predictions)
    accuracy = accuracy_score(y_test, final_predictions)
    print(f"Overall Accuracy: {accuracy}")
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
# Example usage:
# vote4auth(data_scaled, labels)

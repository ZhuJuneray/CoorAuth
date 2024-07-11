from datetime import datetime
import numpy as np
from sklearn.utils import shuffle
from ml4auth import knn_multi_kfolds, knn_multi, knn_binary, knn_binary_kfolds
from ml4auth import svm_multi_kfolds, svm_multi, svm_binary, svm_binary_kfolds
from ml4auth import rf_multi, rf_multi_kfolds, rf_binary_kfolds
from dl4auth import mlp_multi_kfolds, mlp_binary_kfolds, mlp_binary, mlp_multi, lstm_binary, lstm_binary_kfolds, lstm_multi_kfolds
import os, json, re, sys, time
from data_preprocess import data_augment_and_label, read_data_latter_data_json
from vote4auth import vote4auth

def calculate_binary_frr_far(confusion_matrix):
    TP = confusion_matrix[1, 1]
    FN = confusion_matrix[1, 0]
    FP = confusion_matrix[0, 1]
    TN = confusion_matrix[0, 0]

    # 计算 FRR
    FRR = FN / (TP + FN)

    # 计算 FAR
    FAR = FP / (FP + TN)
    print(f"FRR: {FRR} FAR: {FAR}")
    return FRR, FAR



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


################################################################ main
def main():
    current_working_directory = r"/Users/ray/Documents/VR_Authentication"
    os.chdir(current_working_directory)  # cwd的绝对路径
    positive_label = ['14']  # 正样本, 如果要测latter, 则正样本需要是latter data里面的
    model = 'head+eye'  # model
    lstm_model = 'lstm_head+eye'  # lstm model, 此举是因为lstm的数据处理方式不同
    n_split = 4  # k fold
    noise_level = 0.3  # noise level
    augmentation_time = 1  # guassion_aug
    size_list = [3]  # list of size
    all_pin_list = [1]  # pin list
    test_size = 0.3

    # 0120 update
    # json_name = 'data_condition.json'
    # json_name = 'data_given_3days.json'
    # json_name = 'data_own_3days.json'
    json_name = '20240515_test.json'

    # print("")

    for poslabel in positive_label:

        positive_label = poslabel

        print(f"************ positive_label: {positive_label} ************ ")
        print(f"model:{model}, augmentation_time: {augmentation_time}")
        print(f"studytype_users_dates_range: {read_data_latter_data_json(current_working_directory+'/src/'+json_name)[0]}")
        print(f"n_split: {n_split}, noise_level: {noise_level}")
        print(f"segmantation num: 4")#这个在def data_augment_and_label的reshape 里面写死了，也是extract_features的slice_num参数

        for pin in all_pin_list:
            pin_list = [pin]
            print(f"----------------pin_list: {pin_list}----------------")
            # 1.1update augment_time表示增强为原来数量的多少倍，如果留空则为默认值1，即全部为原始数据
            data_scaled, labels, binary_labels, data_scaled_lstm = data_augment_and_label(
                default_authentications_per_person=9, rotdir=os.path.join(os.getcwd(), "data/"), positive_label=positive_label,
                model=model, lstm_model=lstm_model, studytype_users_dates_range=read_data_latter_data_json(current_working_directory+'/src/'+json_name)[0],
                size_list=size_list, pin_list=pin_list,
                noise_level=noise_level, augment_time=augmentation_time)

            print(f"labels:{labels}")
            print(f"binary_labels:{binary_labels}")
            print(f"binary_labels_augmented:{binary_labels_augmented}")
            # print(f"data_scaled:{data_scaled.shape}")
            # 按照标准的ratio做增强
            print("")
            # print("data augment for multiclass")

            # print("---------knn_binary_kfold------------")
            # knn_binary_kfolds(data_scaled=data_scaled, binary_labels=binary_labels,
            #                 n_splits=n_split)

            # print("---------knn_multi_kfold------------")
            # knn_multi_kfolds(data_scaled=data_scaled, labels=labels,
            #                 n_splits=n_split)
            #
            # # print("----------svm_binary_kfold------------")
            # # svm_binary_kfolds(data_scaled=data_scaled, binary_labels=binary_labels,
            # #                 n_splits=n_split)
            #
            # print("-----------svm_multi_kfold------------")
            # svm_multi_kfolds(data_scaled=data_scaled, labels=labels,
            #                 n_splits=n_split)
            #
            # # print("------------mlp_binary_kfold------------")
            # # mlp_binary_kfolds(data_scaled=data_scaled, binary_labels=binary_labels,
            # #                 n_splits=n_split)
            #
            # print("------------mlp_multi_kfold------------")
            # mlp_multi_kfolds(data_scaled=data_scaled, labels=labels,
            #                 n_splits=n_split)
            n_segments = 4
            X_split = split_time_series(data_scaled, n_segments)

            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.metrics import accuracy_score
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import confusion_matrix

            # here whether we choose binary labels
            X_train, X_test, y_train, y_test = train_test_split(X_split, labels, test_size=test_size, stratify=labels)

            print("X_shape", X_train.shape, X_test.shape)
            # 对每个段独立训练分类器
            classifiers = []
            scalers = []
            for segment in range(n_segments):
                # 提取训练数据的当前段
                X_train_segment = X_train[:, segment, :]
                # 训练分类器
                classifier = RandomForestClassifier(n_estimators=100)
                # classifier = KNeighborsClassifier(n_neighbors=3)
                scaler = StandardScaler()
                X_train_segment = scaler.fit_transform(X_train_segment)
                scalers.append(scaler)
                classifier.fit(X_train_segment, y_train)
                classifiers.append(classifier)

            for segment in range(n_segments):
                X_test[:, segment, :] = scalers[segment].transform(X_test[:, segment, :])

            # 对测试集进行投票
            final_predictions = []
            class_list = classifiers[0].classes_

            for sample in X_test:
                segment_predictions = [classifier.predict(sample[segment, :].reshape(1, -1))[0] for segment, classifier
                                       in enumerate(classifiers)]
                # print("segment", segment_predictions)
                final_prediction = max(set(segment_predictions), key=segment_predictions.count)  # 硬投票
                final_predictions.append(final_prediction)

            # print("final_p", final_predictions, "y", y_test)
            # 评估整体性能
            conf_matrix = confusion_matrix(y_test, final_predictions)
            accuracy = accuracy_score(y_test, final_predictions)
            print(f"Overall Accuracy: {accuracy}")
            print(f"confusion matrix: \n {conf_matrix}")
            calculate_binary_frr_far(conf_matrix)
            # frr_list = []
            # far_list = []
            #
            # from selection_calculate import calculate_frr_far
            # for i in range(len(class_list)):
            #     frr, far = calculate_frr_far(conf_matrix, i)
            #     frr_list.append(frr)
            #     far_list.append(far)
            #
            # frr_mean = np.mean(frr_list)
            # far_mean = np.mean(far_list)
            # frr_std = np.std(frr_list) / np.sqrt(len(frr_list))
            # far_std = np.std(far_list) / np.sqrt(len(far_list))
            # print(f"Frr: {frr_mean}, Far: {far_mean}, StdFrr: {frr_std}, StdFar: {far_std}")

            ## with probability
            # 对测试集进行概率投票
            final_predictions = []
            threshold = 0.2  # 设置合理的阈值
            print(f"class_list: {class_list}")
            print(f"Threshold: {threshold}")
            for sample in X_test:
                segment_proba = [classifier.predict_proba(sample[segment, :].reshape(1, -1))[0] for segment, classifier
                                 in enumerate(classifiers)]
                avg_proba = np.mean(segment_proba, axis=0)  # 对每个类的概率进行平均
                # print("segment_proba", segment_proba)
                print("avg_proba", avg_proba)

                if max(avg_proba) > threshold:
                    # final_prediction = np.argmax(avg_proba)  # 选最大的
                    final_prediction = np.random.choice(np.flatnonzero(avg_proba > threshold))  # 随机选一个超过阈值的
                else:
                    final_prediction = np.random.choice(np.flatnonzero(avg_proba == avg_proba.max()))  # 置信度低时随机选择

                final_predictions.append(class_list[final_prediction])

            # 评估整体性能
            conf_matrix = confusion_matrix(y_test, final_predictions)
            accuracy = accuracy_score(y_test, final_predictions)
            print(f"Overall Accuracy: {accuracy}")
            print(f"confusion matrix: \n {conf_matrix}")
            calculate_binary_frr_far(conf_matrix)
            # frr_list = []
            # far_list = []
            #
            # from selection_calculate import calculate_frr_far
            # for i in range(len(class_list)):
            #     frr, far = calculate_frr_far(conf_matrix, i)
            #     frr_list.append(frr)
            #     far_list.append(far)
            #
            # frr_mean = np.mean(frr_list)
            # far_mean = np.mean(far_list)
            # frr_std = np.std(frr_list) / np.sqrt(len(frr_list))
            # far_std = np.std(far_list) / np.sqrt(len(far_list))
            # print(f"Frr: {frr_mean}, Far: {far_mean}, StdFrr: {frr_std}, StdFar: {far_std}")

            # vote4auth(data_scaled=data_scaled, labels=labels, test_size=test_size, n_segments=n_segments)
            # 数据增强后的数据和标签跑模型
            # print("")
            # # print("data augment for binary")
            # #
            # print("---------knn_binary_kfold------------")
            # knn_binary_kfolds(data_scaled=scaled_data_augmented, binary_labels=binary_labels_augmented,
            #                 n_splits=n_split)
            #
            # print("----------svm_binary_kfold------------")
            # svm_binary_kfolds(data_scaled=scaled_data_augmented, binary_labels=binary_labels_augmented,
            #                 n_splits=n_split)
            #
            # print("------------mlp_binary_kfold------------")
            # mlp_binary_kfolds(data_scaled=scaled_data_augmented, binary_labels=binary_labels_augmented,
            #                 n_splits=n_split)

            ################################ 随时间推移重新检验部分
            # 日后新采的数据属性
            # default_latter_auth_per_person = 4  # 每人采集次数
            # latter_positive_label = positive_label  # 正样本, 与之前是一致的
            #
            # latter_data_scaled, latter_labels, latter_binary_labels, _, _ = data_augment_and_label(
            #     default_authentications_per_person=default_latter_auth_per_person, rotdir=os.path.join(os.getcwd(), "data/"),
            #     positive_label=latter_positive_label, model=model,
            #     studytype_users_dates_range=read_data_latter_data_json(current_working_directory+'/src/'+json_name)[1],
            #     size_list=size_list, pin_list=pin_list, noise_level=noise_level)
            #
            # print("")
            # print(f"latter_data_scaled: {latter_data_scaled.shape}")
            # print("")
            #
            # # latter_data_scaled, latter_labels, latter_binary_labels = shuffle(latter_data_scaled, latter_labels, latter_binary_labels)
            # # print("--------knn_binary------------")
            # # knn_binary(data_scaled=scaled_data_augmented, binary_labels=binary_labels_augmented,
            # #            latter_data_scaled=latter_data_scaled, latter_labels=latter_binary_labels, test_size=test_size)
            #
            # print("---------knn_multi------------")
            # knn_multi(data_scaled=data_scaled, labels=labels, latter_data_scaled=latter_data_scaled,
            #           latter_labels=latter_labels, test_size=test_size)
            #
            # # print("---------svm_binary------------")
            # # svm_binary(data_scaled=scaled_data_augmented, binary_labels=binary_labels_augmented,
            # #            latter_data_scaled=latter_data_scaled, latter_labels=latter_binary_labels, test_size=test_size)
            #
            # print("---------svm_multi------------")
            # svm_multi(data_scaled=data_scaled, labels=labels,
            #           latter_data_scaled=None, latter_labels=None, test_size=test_size)
            #
            # # print("---------mlp_binary------------")
            # # mlp_binary(data_scaled=scaled_data_augmented, binary_labels=binary_labels_augmented,
            # #         latter_data_scaled=latter_data_scaled, latter_labels=latter_binary_labels, test_size=test_size)
            #
            # print("-----------mlp_multi------------")
            # mlp_multi(data_scaled=data_scaled, labels=labels,
            #         latter_data_scaled=None, latter_labels=None, test_size=test_size)

            # print("---------rf_multi------------")
            # rf_multi(data_scaled=data_scaled, labels=labels, test_size=test_size)

            default_latter_auth_per_person = 4  # 每人采集次数
            latter_positive_label = positive_label  # 正样本, 与之前是一致的
            #
            latter_data_scaled, latter_labels, latter_binary_labels, latter_data_scaled_lstm= data_augment_and_label(
                default_authentications_per_person=default_latter_auth_per_person, rotdir=os.path.join(os.getcwd(), "data/"),
                positive_label=latter_positive_label, model=model, lstm_model=lstm_model,
                studytype_users_dates_range=read_data_latter_data_json(current_working_directory+'/src/'+json_name)[1],
                size_list=size_list, pin_list=pin_list, noise_level=noise_level)
            #
            print("")
            print(f"latter_data_scaled: {latter_data_scaled.shape}")
            # print(f"latter_labels: {latter_labels}")
            print("")
            #
            latter_data_scaled, latter_labels, latter_binary_labels = shuffle(latter_data_scaled, latter_labels, latter_binary_labels)
            # print("--------knn_binary------------")
            # knn_binary(data_scaled=scaled_data_augmented, binary_labels=binary_labels_augmented,
            #            latter_data_scaled=latter_data_scaled, latter_labels=latter_binary_labels, test_size=test_size)
            
            # print("---------knn_multi------------")
            # knn_multi(data_scaled=data_scaled, labels=labels, latter_data_scaled=latter_data_scaled,
            #           latter_labels=latter_labels, test_size=test_size)
            # #

            print("----------rf_binary_kfold------------")
            rf_binary_kfolds(data_scaled=data_scaled, labels=binary_labels, n_splits=n_split, latter_data_scaled=latter_data_scaled, latter_labels=latter_binary_labels)

            print("----------rf_multi_kfold------------")
            rf_multi_kfolds(data_scaled=data_scaled, labels=labels, n_splits=n_split, latter_data_scaled=latter_data_scaled, latter_labels=latter_labels)

            # print("----------lstm_binary_kfolds------------")
            # lstm_binary_kfolds(data_scaled=data_scaled_lstm, binary_labels=binary_labels, epochs=20, batch_size=4, latter_data_scaled=latter_data_scaled_lstm, latter_labels=latter_binary_labels)

            # print("----------lstm_multi_kfolds------------")
            # lstm_multi_kfolds(data_scaled=data_scaled_lstm, labels=labels, epochs=20, batch_size=4, latter_data_scaled=latter_data_scaled_lstm, latter_labels=latter_labels)



if __name__ == "__main__":
    current_datetime = datetime.now()
    filename = "output" + str(current_datetime).replace(' ', '-').replace('.', '-').replace(':', '-') + ".txt"
    with open(filename, 'w') as file:  # 将print内容保存到文件
        # 保存当前的标准输出
        original_stdout = sys.stdout
        # 将标准输出重定向到文件
        sys.stdout = file
        start_time = time.time()
        main()
        end_time = time.time()
        run_time = end_time - start_time
        print(f"程序运行时间：{run_time}秒")
        # 恢复原来的标准输出
        sys.stdout = original_stdout

    # main() # 用于在终端输出print内容

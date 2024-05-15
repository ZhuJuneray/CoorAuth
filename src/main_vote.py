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
    positive_label = ['4']  # 正样本
    model = 'head+eye'  # model
    n_split = 4  # k fold
    noise_level = 0.3  # 高斯噪声做数据增强的强度
    augmentation_time = 10  # 数据增强数据量的倍数
    size_list = [3]  # list of size
    all_pin_list = [1]  # pin list
    test_size = 0.3
    n_segments = 1

    # 0120 update
    # json_name = 'data_condition.json'
    # json_name = 'data_given_3days.json'
    # json_name = 'data_own_3days.json'
    json_name = 'data_split_trainset.json'

    for poslabel in positive_label:

        positive_label = [poslabel]

        print(f"************ positive_label: {positive_label} ************ ")
        print(f"model:{model}, augmentation_time: {augmentation_time}")
        print(f"studytype_users_dates_range: {read_data_latter_data_json(current_working_directory+'/src/'+json_name)[0]}")
        print(f"n_split: {n_split}, noise_level: {noise_level}")
        print(f"segmantation num: 4")#这个在def data_augment_and_label的reshape 里面写死了，也是extract_features的slice_num参数

        for pin in all_pin_list:
            pin_list = [pin]
            print(f"----------------pin_list: {pin_list}----------------")
            # 1.1update augment_time表示增强为原来数量的多少倍，如果留空则为默认值1，即全部为原始数据
            data_scaled, labels, binary_labels, scaled_data_augmented, binary_labels_augmented, data_scaled_lstm, data_scaled_augmented_lstm = data_augment_and_label(
                default_authentications_per_person=9, rotdir=os.path.join(os.getcwd(), "data/"), positive_label=positive_label,
                model=model, studytype_users_dates_range=read_data_latter_data_json(current_working_directory+'/src/'+json_name)[0],
                size_list=size_list, pin_list=pin_list,
                noise_level=noise_level, augment_time=augmentation_time)

            # print(f"labels:{labels}")
            # print(f"binary_labels:{binary_labels}")
            # print(f"binary_labels_augmented:{binary_labels_augmented}")
            print(f"data_scaled:{data_scaled.shape}")
            print(f"data_scaled_lstm:{data_scaled_lstm.shape}")
            # np.savetxt("data_scaled.txt", np.array(data_scaled))
            
            # print(f"scaler_origin:{scaler_origin.mean_}, scaler_augment:{scaler_augment.mean_}")
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

            # 4.28
            # n_segments = 4
            # X_split = split_time_series(data_scaled, n_segments)

            # from sklearn.model_selection import train_test_split
            # from sklearn.ensemble import RandomForestClassifier
            # from sklearn.neighbors import KNeighborsClassifier
            # from sklearn.metrics import accuracy_score
            # from sklearn.preprocessing import StandardScaler
            # from sklearn.metrics import confusion_matrix

            # X_train, X_test, y_train, y_test = train_test_split(X_split, labels, test_size=test_size, stratify=labels)

            # print("X_shape", X_train.shape, X_test.shape)
            # # 对每个段独立训练分类器
            # classifiers = []
            # scalers = []
            # for segment in range(n_segments):
            #     # 提取训练数据的当前段
            #     X_train_segment = X_train[:, segment, :]
            #     # 训练分类器
            #     classifier = RandomForestClassifier(n_estimators=100)
            #     # classifier = KNeighborsClassifier(n_neighbors=3)
            #     scaler = StandardScaler()
            #     X_train_segment = scaler.fit_transform(X_train_segment)
            #     scalers.append(scaler)
            #     classifier.fit(X_train_segment, y_train)
            #     classifiers.append(classifier)

            # for segment in range(n_segments):
            #     X_test[:, segment, :] = scalers[segment].transform(X_test[:, segment, :])

            # # 对测试集进行投票
            # final_predictions = []
            # for sample in X_test:
            #     segment_predictions = [classifier.predict(sample[segment, :].reshape(1, -1))[0] for segment, classifier
            #                            in enumerate(classifiers)]
            #     final_prediction = max(set(segment_predictions), key=segment_predictions.count)  # 硬投票
            #     print("segment", segment_predictions, "final", final_prediction)
            #     final_predictions.append(final_prediction)

            # print("final_p", final_predictions, "y", y_test)
            # # 评估整体性能
            # conf_matrix = confusion_matrix(y_test, final_predictions)
            # accuracy = accuracy_score(y_test, final_predictions)
            # print(f"Overall Accuracy: {accuracy}")
            # 4.28

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

            # print("----------lstm_binary_kfolds------------")
            # lstm_binary_kfolds(data_scaled=data_scaled_augmented_lstm, binary_labels=binary_labels, epochs=20, batch_size=4)

            # print("----------lstm_multi_kfolds------------")
            # lstm_multi_kfolds(data_scaled=data_scaled_augmented_lstm, labels=labels, epochs=20, batch_size=4)

            print("----------rf_binary_kfold------------")
            rf_binary_kfolds(data_scaled=data_scaled, labels=binary_labels, n_splits=n_split)

            print("----------rf_multi_kfold------------")
            rf_multi_kfolds(data_scaled=data_scaled, labels=labels, n_splits=n_split)

            default_latter_auth_per_person = 4  # 每人采集次数
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

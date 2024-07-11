from datetime import datetime
import numpy as np
from sklearn.utils import shuffle
from ml4auth import knn_multi_kfolds, knn_multi, knn_binary, knn_binary_kfolds
from ml4auth import svm_multi_kfolds, svm_multi, svm_binary, svm_binary_kfolds
from ml4auth import rf_multi
from dl4auth import mlp_multi_kfolds, mlp_binary_kfolds, mlp_binary, mlp_multi
import os, json, re, sys, time
from data_preprocess import data_augment_and_label, read_data_latter_data_json


################################################################ main
def main():
    current_working_directory = r"D:\pycharm\srt_vr_auth"
    os.chdir(current_working_directory)  # cwd的绝对路径
    positive_label = ['14']  # 正样本
    model = 'head+eye'  # model
    n_split = 4  # k fold
    noise_level = 0.3  # noise level
    augmentation_time = 1  # 高斯噪声做数据增强的倍数
    size_list = [3]  # list of size
    all_pin_list = [1]  # pin list
    # 1.25更新
    test_size_list = [0.2]
    data_scaled_list = []
    labels_list = []
    # 0120 update
    # json_name = 'data_condition.json'
    # json_name = 'data_given_3days.json'
    # json_name = 'data_own_3days.json'
    json_name = 'data_split_trainset.json'

    print(f"model:{model}, augmentation_time: {augmentation_time}")
    # print(f"studytype_users_dates_range: {read_data_latter_data_json(current_working_directory + '/src/' + json_name)[0]}")

    # 1.25更新，先计算pin，将所有数据存储到list，这样节省时间
    for pin in range(len(all_pin_list)):
        pin_list = [all_pin_list[pin]]
        # 1.1update augment_time表示增强为原来数量的多少倍，如果留空则为默认值1，即全部为原始数据
        data_scaled, labels, binary_labels, scaled_data_augmented, binary_labels_augmented = data_augment_and_label(
            default_authentications_per_person=9, rotdir=os.path.join(os.getcwd(), "data/"),
            positive_label=positive_label, model=model,
            studytype_users_dates_range=read_data_latter_data_json(current_working_directory + '/src/' + json_name)[0],
            size_list=size_list, pin_list=pin_list, noise_level=noise_level, augment_time=augmentation_time)

        data_scaled_list.append(data_scaled)
        labels_list.append(labels)

    for test_size in test_size_list:

        print(f"test size: {test_size} ******************* ")

        for pin in range(len(all_pin_list)):
            pin_list = [all_pin_list[pin]]
            print(f"pin_list: {pin_list}----------------------")

            data_scaled = data_scaled_list[pin]
            labels = labels_list[pin]
            # print(f"labels:{labels}")
            # print(f"binary_labels:{binary_labels}")
            # print(f"binary_labels_augmented:{binary_labels_augmented}")
            # print(f"data_scaled:{data_scaled.shape}")
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
            default_latter_auth_per_person = 4  # 每人采集次数
            latter_positive_label = positive_label  # 正样本, 与之前是一致的

            latter_data_scaled, latter_labels, latter_binary_labels, _, _ = data_augment_and_label(
                default_authentications_per_person=default_latter_auth_per_person, rotdir=os.path.join(os.getcwd(), "data/"),
                positive_label=latter_positive_label, model=model,
                studytype_users_dates_range=read_data_latter_data_json(current_working_directory+'/src/'+json_name)[1],
                size_list=size_list, pin_list=pin_list, noise_level=noise_level)

            print("")
            print(f"latter_data_scaled: {latter_data_scaled.shape}")
            print("")
            #
            # latter_data_scaled, latter_labels, latter_binary_labels = shuffle(latter_data_scaled, latter_labels, latter_binary_labels)
            # print("--------knn_binary------------")
            # knn_binary(data_scaled=scaled_data_augmented, binary_labels=binary_labels_augmented,
            #            latter_data_scaled=latter_data_scaled, latter_labels=latter_binary_labels, test_size=test_size)

            print("---------knn_multi------------")
            knn_multi(data_scaled=data_scaled, labels=labels, latter_data_scaled=None,
                      latter_labels=None, test_size=test_size)

            # print("---------svm_binary------------")
            # svm_binary(data_scaled=scaled_data_augmented, binary_labels=binary_labels_augmented,
            #            latter_data_scaled=latter_data_scaled, latter_labels=latter_binary_labels, test_size=test_size)

            print("---------svm_multi------------")
            svm_multi(data_scaled=data_scaled, labels=labels,
                      latter_data_scaled=None, latter_labels=None, test_size=test_size)

            # print("---------mlp_binary------------")
            # mlp_binary(data_scaled=scaled_data_augmented, binary_labels=binary_labels_augmented,
            #         latter_data_scaled=latter_data_scaled, latter_labels=latter_binary_labels, test_size=test_size)

            print("-----------mlp_multi------------")
            mlp_multi(data_scaled=data_scaled, labels=labels,
                      latter_data_scaled=None, latter_labels=None, test_size=test_size)

            print("-----------rf_multi------------")
            rf_multi(data_scaled=data_scaled, labels=labels,
                      latter_data_scaled=latter_data_scaled, latter_labels=latter_labels, test_size=test_size)


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
        print("--------")
        print("******** \n")
        print(f"程序运行时间：{run_time}秒")
        # 恢复原来的标准输出
        sys.stdout = original_stdout

    # main() # 用于在终端输出print内容

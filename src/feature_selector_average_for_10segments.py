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
from sklearn.feature_selection import SelectKBest, f_classif


def smooth_data(arr, window_parameter=9, polyorder_parameter=2):
    arr_smoothed = savgol_filter(arr, window_length=window_parameter, polyorder=polyorder_parameter)
    return arr_smoothed


def read_data_name_from_json(filepath=(os.getcwd()+"/src/data.json")):
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
        # noise_scale = noise_level * positive_features_to_augment.std()  # 调整噪声水平
        # gaussian_noise = np.random.normal(0, noise_scale, positive_features_to_augment.shape)
        # positive_features_noisy = positive_features_to_augment + gaussian_noise

        # 将增强的样本合并回原始数据集
        result_array_augmented = np.concatenate((result_array, positive_features_to_augment), axis=0)
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



################################################################ main
def main():
    # os.chdir('D:\pycharm\srt_vr_auth') # cwd的绝对路径
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

    print("特征选择：")
    X = data_scaled
    print("X:", X)
    y = binary_labels
    

    selector = SelectKBest(score_func=f_classif, k='all')  # k='all'意味着选择所有特征
    selector.fit(X, y)

    # 获取各特征的Fisher得分
    scores = selector.scores_

    scores_averaged = []
    sum = 0
    count = 0

    segment_len = 10 # 切段数量

    for score in scores:
        sum += score
        count += 1
        if count == segment_len:
            score_averaged = sum / segment_len
            scores_averaged.append(score_averaged)
            sum = 0
            count = 0

    # 打印特征得分
    print(f"Feature scores len:{len(scores_averaged)}")
    print("Feature scores:", scores_averaged)

    

    # # 可能你会想根据得分选择特征
    # num_features_selected = 200  # 你想选择最好的特征的数量
    # top_indices = np.argsort(scores)[::-1]
    # print(f"original top_indices:{top_indices}")
    # print(f"original top scores:{scores[top_indices]}")

    type_name_list = []
    if model == 'head':
        type_name_list = ["initial_d1_feat", "initial_d2_feat", "initial_d3_feat", "initial_d4_feat", "initial_v1_feat",
                           "initial_v2_feat", "initial_v3_feat", "velocity_d1_feat", "velocity_d2_feat", "velocity_d3_feat",
                             "velocity_d4_feat", "velocity_v1_feat", "velocity_v2_feat", "velocity_v3_feat"]
    elif model == "eye":
        type_name_list = ["d1_el_feat", "d2_el_feat", "d3_el_feat", "d4_el_feat", "d1_er_feat", "d2_er_feat", "d3_er_feat", "d4_er_feat", ]
    elif model == "head+eye":
        type_name_list = ["d1_feat", "d2_feat", "d3_feat", "d4_feat", "v1_feat", "v2_feat", "v3_feat", "d1_el_feat", "d2_el_feat", "d3_el_feat", "d4_el_feat", "d1_er_feat", "d2_er_feat", "d3_er_feat", "d4_er_feat", ]
    elif model == "diff":
        type_name_list = ["dy_el_feat", "dp_el_feat", "dr_el_feat"]
    elif model == "eye+diff":
        type = ["dy_el_feat", "dp_el_feat", "dr_el_feat", "d1_el_feat", "d2_el_feat", "d3_el_feat", "d4_el_feat", "d1_er_feat", "d2_er_feat", "d3_er_feat", "d4_er_feat"]
    
    feature_name_list = ["features_mean", "features_max", "features_min", "features_var", "features_median"
                         , "features_rms", "features_std", "features_mad" ,"features_iqr", "features_mc", "features_wamp", "features_ssc"]

    start = 0
    end = 12
    import matplotlib.pyplot as plt
    fig1 = plt.figure(figsize=(40, 8))
    for _type in type_name_list:
        print(f"type:{_type}")
        
        type_scores_averaged = scores_averaged[start:end]
        type_top_indices = np.argsort(type_scores_averaged)[::-1] #把type_scores里的元素从大到小排序，返回索引
        for i in type_top_indices: 
            if not np.isnan(type_scores_averaged[i]):
                print("    statistical feature: ", feature_name_list[i%12] , type_scores_averaged[i])

        type_feature_names = [feature_name_list[i % 12] for i in type_top_indices if not np.isnan(type_scores_averaged[i])]
        type_feature_scores = [type_scores_averaged[i] for i in type_top_indices if not np.isnan(type_scores_averaged[i])]

        # 创建一个子图
        ax1 = fig1.add_subplot(1, len(type_name_list), type_name_list.index(_type) + 1)
        ax1.bar(type_feature_names, type_feature_scores)
        ax1.set_xlabel("Features")
        ax1.set_ylabel("Averaged Scores")
        ax1.set_title(f"{_type}")
        ax1.tick_params(axis='x', rotation=90)

        start += 12
        end += 12
    
    # 保存整体图形到一个PNG文件
    plt.tight_layout()
    plt.savefig("result/fisher_score/all_fisher_score_by_type.png")

    start = 0
    end = 12

    for _type in type_name_list:

        type_scores_averaged = scores_averaged[start:end]
        type_top_indices = np.argsort(type_scores_averaged)[::-1]

        type_feature_names = [feature_name_list[i % 12] for i in type_top_indices if not np.isnan(type_scores_averaged[i])]
        type_feature_scores = [type_scores_averaged[i] for i in type_top_indices if not np.isnan(type_scores_averaged[i])]

        # 创建一个新的图形和子图
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)

        ax.bar(type_feature_names, type_feature_scores)
        ax.set_xlabel("Features")
        ax.set_ylabel("Averaged Scores")
        ax.set_title(f"Type: {_type} - Averaged Scores for Top Features")
        ax.tick_params(axis='x', rotation=90)

        # 保存图形到不同的文件名
        plt.tight_layout()
        plt.savefig(f"result/fisher_score/fisher_score_for_{_type}.png")

        start += 12
        end += 12


    # 获取排序后的索引
    nan_indices = np.where(np.isnan(scores_averaged))
    # # 从 scores 数组中去掉空值及其对应的索引
    nan_deleted_scores_averaged = np.delete(scores_averaged, nan_indices)
    # top_indices = np.argsort(scores)[-num_features_selected:]
    # print("nan deleted top_indices:", top_indices)
    # print(f"nan deleted top scores:{scores[top_indices]} ")
    sorted_indices = np.argsort(nan_deleted_scores_averaged)[::-1]

    print(f"sorted_indices:{sorted_indices}")

    # 获取排序后的特征名称和分数
    sorted_feature_names = [feature_name_list[i % 12] + " and " + type_name_list[i % 14] for i in sorted_indices]
    sorted_scores = [nan_deleted_scores_averaged[i] for i in sorted_indices]
    print(f"sorted_feature_names:{sorted_feature_names}")
    print(f"sorted_scores:{sorted_scores}")

    # 创建条形图
    plt.figure(figsize=(40, 8))
    plt.bar(sorted_feature_names, sorted_scores)
    plt.xlabel("Features")
    plt.ylabel("Averaged Scores")
    plt.title("Averaged Scores for Features (Sorted by Score)")
    plt.xticks(rotation=90)
    plt.tight_layout()

    # 保存图形到文件
    plt.savefig(f"result/fisher_score/all_sorted_fisher_scores.png")


    scores_for_feature = []
    for i in range(len(feature_name_list)):
        # 初始化结果列表
        averaged_list = []
        
        list_to_average = [ scores_averaged[j] for j in range(i, len(scores_averaged), len(feature_name_list))]
        scores_for_feature.append(np.array(list_to_average).mean())
    
    top_indices_for_feature = np.argsort(scores_for_feature)[::-1]

    # 创建条形图
    plt.figure(figsize=(12, 6))
    plt.bar([feature_name_list[i] for i in top_indices_for_feature], [scores_for_feature[i] for i in top_indices_for_feature])
    plt.xlabel("Features")
    plt.ylabel("Averaged Scores")
    plt.title("Averaged Scores for Features (Sorted by Score)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"result/fisher_score/all_sorted_fisher_scores_averaged_for_features.png")


    




    


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

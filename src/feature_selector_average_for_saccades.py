from ml4auth import knn_multi_kfolds, knn_multi, knn_binary, knn_binary_kfolds
from ml4auth import svm_multi_kfolds, svm_multi, svm_binary, svm_binary_kfolds
from dl4auth import mlp_multi_kfolds, mlp_binary_kfolds, mlp_binary, mlp_multi
import os, json, re, sys
from data_preprocess import data_augment_and_label, read_data_latter_data_json
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
import matplotlib.pyplot as plt


################################################################ main
def main():
    current_working_directory = "/Users/ray/Documents/VR_Authentication"
    os.chdir(current_working_directory)  # cwd的绝对路径
    positive_label = ['1']  # 正样本
    model = 'head+eye'  # model
    n_split = 3  # k fold
    noise_level = 0.3  # noise level
    augmentation_time = 1  # 高斯噪声做数据增强的倍数
    size_list = [3]  # list of size
    all_pin_list = [1]  # pin list
    # 0120 update
    json_name = 'data_feature_selection.json'
    # json_name = 'data_condition.json'
    print(f"positive_label: {positive_label}, model: {model}, augmentation_time: {augmentation_time}")
    print(f"model:{model}, size_list: {size_list}")
    print(f"studytype_users_dates_range: {read_data_latter_data_json(current_working_directory+'/src/'+json_name)[0]}")

    # 1.3update parameters for models
    svm_kernel = 'linear'
    knn_neighbors = 3
    mlp_hiddenlayer = (256,)
    mlp_iteration = 200
    for pin in all_pin_list:
        pin_list = [pin]
        print(f"----------------pin_list: {pin_list}----------------")
        # 1.1update augment_time表示增强为原来数量的多少倍，如果留空则为默认值1，即全部为原始数据
        data_scaled, labels, binary_labels, scaled_data_augmented, binary_labels_augmented = data_augment_and_label(
            default_authentications_per_person=9, rotdir=os.path.join(os.getcwd(), "data/"), positive_label=positive_label,
            model=model, studytype_users_dates_range=read_data_latter_data_json(current_working_directory+'/src/'+json_name)[0],
            size_list=size_list, pin_list=pin_list,
            noise_level=noise_level, augment_time=augmentation_time)

        print(f"labels:{labels}")
        print(f"binary_labels:{binary_labels}")
        print(f"binary_labels_augmented:{binary_labels_augmented}")
        print(f"data_scaled:{data_scaled.shape}")
        # print(f"scaler_origin:{scaler_origin.mean_}, scaler_augment:{scaler_augment.mean_}")
        # 原数据和标签跑机器学习模型
        print("")
        print("data augment for multiclass")

        print("特征选择：")
        X = data_scaled
        print("X:", X)
        y = labels
        

        selector = SelectKBest(score_func=f_classif, k='all')  # k='all'意味着选择所有特征
        selector.fit(X, y)

        # 获取各特征的Fisher得分
        scores = selector.scores_

        scores_averaged = []
        _sum = 0
        count = 0

        segment_len = 10 # 切段数量

        max_or_average = "average"
        # 取10段的平均，用scores_averaged表示
        for score in scores:
            _sum += score
            count += 1
            if count == 1:
                score_averaged = _sum / segment_len
                scores_averaged.append(score_averaged)
            if count == segment_len:
                _sum = 0
                count = 0

        # 取最大值，也用scores_averaged表示
        # max_score_in_segment = float('-inf')  # 用于存储每个段内的最大分数
        # for score in scores:
        #     count += 1
        #     if score > max_score_in_segment:
        #         max_score_in_segment = score  # 更新段内最大分数

        #     if count == segment_len:
        #         scores_averaged.append(max_score_in_segment)  # 将段内最大分数添加到列表中
        #         max_score_in_segment = float('-inf')  # 重置段内最大分数
        #         count = 0

            

        # 打印特征得分
        print(f"Note: In this case we choose ")
        print(f"Averaged feature scores len:{len(scores_averaged)}")
        print("Averaged feature scores:", scores_averaged)

        

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
            type_name_list = ["initial_d1_feat", "second_d1_feat", "initial_d2_feat", "second_d2_feat",
                               "initial_d3_feat", "second_d3_feat", "initial_d4_feat", "second_d4_feat",
                               "initial_v1_feat", "second_v1_feat", "initial_v2_feat", "second_v2_feat",
                                 "initial_v3_feat","second_v3_feat", "initial_d1_el_feat", "second_d1_el_feat",
                                 "initial_d2_el_feat", "second_d2_el_feat", "initial_d3_el_feat", "second_d3_el_feat",
                                 "initial_d4_el_feat", "second_d4_el_feat", "initial_d1_er_feat", "second_d1_er_feat",
                                   "initial_d2_er_feat", "second_d2_er_feat", "initial_d3_er_feat", "second_d3_er_feat",
                                   "initial_d4_er_feat", "second_d4_er_feat"] # len: 30
                                  
        elif model == "diff":
            type_name_list = ["dy_el_feat", "dp_el_feat", "dr_el_feat"]
        elif model == "eye+diff":
            type = ["dy_el_feat", "dp_el_feat", "dr_el_feat", "d1_el_feat", "d2_el_feat", "d3_el_feat", "d4_el_feat", "d1_er_feat", "d2_er_feat", "d3_er_feat", "d4_er_feat"]
        
        # 每个type内有以下统计学特征
        feature_name_list = ["features_mean", "features_max", "features_min", "features_var",
                              "features_median", "features_rms", "features_std", "features_mad",
                              "features_iqr", "features_mc", "features_wamp", "features_ssc",
                                "features_kurtosis", "features_skewness" ] # len: 14
        
        saccades_or_fixation = "saccades"
        

        feature_name_num = len(feature_name_list) # feature的数量
        type_name_num = len(type_name_list) # type的数量
       
        # # 画每一个type（feature）内不同统计学特征的score的图
        
        # start = 0
        # end = feature_name_num

        
        # fig1 = plt.figure(figsize=(60, 8))
        # for _type in type_name_list:
        #     print(f"type:{_type}")
            
        #     type_scores_averaged = scores_averaged[start:end]
        #     type_top_indices = np.argsort(type_scores_averaged)[::-1] #把type_scores里的元素从大到小排序，返回索引
        #     # for i in type_top_indices: 
        #     #     if not np.isnan(type_scores_averaged[i]):
        #     #         print("    statistical feature: ", feature_name_list[i%feature_name_num] , type_scores_averaged[i])

        #     type_feature_names = [feature_name_list[i % feature_name_num] for i in type_top_indices if not np.isnan(type_scores_averaged[i])]
        #     type_feature_scores = [type_scores_averaged[i] for i in type_top_indices if not np.isnan(type_scores_averaged[i])]

        #     # 创建一个子图
        #     ax1 = fig1.add_subplot(1, len(type_name_list), type_name_list.index(_type) + 1)
        #     ax1.bar(type_feature_names, type_feature_scores)
        #     ax1.set_xlabel("Features")
        #     ax1.set_ylabel("Averaged Scores")
        #     ax1.set_title(f"{_type}")
        #     ax1.tick_params(axis='x', rotation=90)

        #     start += feature_name_num
        #     end += feature_name_num
        
        # # 保存整体图形到一个PNG文件
        # plt.tight_layout()
        # plt.savefig("result/fisher_score/all_fisher_score_by_type.png")


        # # 相当于把以上all的子图分别保存为一个.png
        # start = 0
        # end = feature_name_num

        # for _type in type_name_list:

        #     type_scores_averaged = scores_averaged[start:end]
        #     type_top_indices = np.argsort(type_scores_averaged)[::-1]

        #     type_feature_names = [feature_name_list[i % feature_name_num] for i in type_top_indices if not np.isnan(type_scores_averaged[i])]
        #     type_feature_scores = [type_scores_averaged[i] for i in type_top_indices if not np.isnan(type_scores_averaged[i])]

        #     # 创建一个新的图形和子图
        #     fig = plt.figure(figsize=(15, 5))
        #     ax = fig.add_subplot(111)

        #     ax.bar(type_feature_names, type_feature_scores)
        #     ax.set_xlabel("Features")
        #     ax.set_ylabel("Averaged Scores")
        #     ax.set_title(f"Type: {_type} - Averaged Scores for Top Features")
        #     ax.tick_params(axis='x', rotation=90)

        #     # 保存图形到不同的文件名
        #     plt.tight_layout()
        #     plt.savefig(f"result/fisher_score/fisher_score_for_{_type}.png")

        #     start += 12
        #     end += 12


        # # 所有score的排名， 每个score是某个type and feature的score，故会有feature_name_num * type_name_num个score
        # nan_indices = np.where(np.isnan(scores_averaged)) # 获取排序后的索引
        # # # 从 scores 数组中去掉空值及其对应的索引
        # nan_deleted_scores_averaged = np.delete(scores_averaged, nan_indices)
        # # top_indices = np.argsort(scores)[-num_features_selected:]
        # # print("nan deleted top_indices:", top_indices)
        # # print(f"nan deleted top scores:{scores[top_indices]} ")
        # sorted_indices = np.argsort(nan_deleted_scores_averaged)[::-1]

        # print(f"sorted_indices:{sorted_indices}")

        # # 获取排序后的特征名称和分数
        # sorted_feature_names = [feature_name_list[i % feature_name_num] + " and " + type_name_list[i % type_name_num] for i in sorted_indices]
        # sorted_scores = [nan_deleted_scores_averaged[i] for i in sorted_indices]
        # print(f"sorted_feature_names:{sorted_feature_names}")
        # print(f"sorted_scores:{sorted_scores}")

        # # 创建条形图
        # plt.figure(figsize=(40, 8))
        # plt.bar(sorted_feature_names, sorted_scores)
        # plt.xlabel("Features")
        # plt.ylabel("Averaged Scores")
        # plt.title("Averaged Scores for Features (Sorted by Score)")
        # plt.xticks(rotation=90)
        # plt.tight_layout()

        # # 保存图形到文件
        # plt.savefig(f"result/fisher_score/all_sorted_fisher_scores.png")



        # 取前100个得分最高的score，统计所有type和feature（分开统计）的分数
        nan_indices = np.where(np.isnan(scores_averaged)) # 获取排序后的索引
        # # 从 scores 数组中去掉空值及其对应的索引
        nan_deleted_scores_averaged = np.delete(scores_averaged, nan_indices)
        sorted_indices = np.argsort(nan_deleted_scores_averaged)[::-1]
        sorted_indices = sorted_indices[:100] # 取前100个得分最高的score的索引

        feature_score = [0] * feature_name_num # 用于存储每个feature的总分数
        type_scores = [0] * type_name_num # 用于存储每个type的总分数
        feature_name_to_display_list = [""] * feature_name_num
        type_name_to_display_list = [""] * type_name_num
        for i in sorted_indices:
            feature_score[i % feature_name_num] += nan_deleted_scores_averaged[i] # 更新feature的总分数
            feature_name_to_display_list[i % feature_name_num] = feature_name_list[i % feature_name_num] # 更新feature的名称
            type_scores[i % type_name_num] += nan_deleted_scores_averaged[i] # 更新type的总分数
            type_name_to_display_list[i % type_name_num] = type_name_list[i % type_name_num]
        
        # 对feature和type的总分数进行排序
        feature_indices = np.argsort(feature_score)[::-1]
        type_indices = np.argsort(type_scores)[::-1]
        feature_score = [feature_score[i] for i in feature_indices]
        feature_name_to_display_list = [feature_name_to_display_list[i] for i in feature_indices]
        type_scores = [type_scores[i] for i in type_indices]
        type_name_to_display_list = [type_name_to_display_list[i] for i in type_indices]
        # 创建条形图
        plt.figure(figsize=(10, 8))
        plt.bar(feature_name_to_display_list, feature_score) 
        plt.xlabel("Features")
        plt.ylabel("Total Scores(added from averaged scores)")
        plt.title(f"Total Scores for each Features (Top100, added from averaged scores, {saccades_or_fixation}, {max_or_average})")
        plt.xticks(rotation=90)
        plt.tight_layout()
        # 保存图形到文件
        plt.savefig(f"result/fisher_score/total_feature_scores_{saccades_or_fixation}_{max_or_average}.png")
        plt.close()

        # 创建条形图
        plt.figure(figsize=(10, 8))
        plt.bar(type_name_to_display_list, type_scores)
        plt.xlabel("Types")
        plt.ylabel("Total Scores(added from averaged scores)")
        plt.title(f"Total Scores for each Types (Top100, added from averaged scores, {saccades_or_fixation}, {max_or_average})")
        plt.xticks(rotation=90)
        plt.tight_layout()
        # 保存图形到文件
        plt.savefig(f"result/fisher_score/total_type_scores_{saccades_or_fixation}_{max_or_average}.png")
        plt.close()



        # # 画通过一个feature的的score的图，每个feature通过其包含的所有type的score求平均
        # scores_for_feature = []
        # for i in range(feature_name_num):
        #     # 初始化结果列表
        #     averaged_list = []
            
        #     list_to_average = [ scores_averaged[j] for j in range(i, len(scores_averaged), len(feature_name_list))]
        #     scores_for_feature.append(np.array(list_to_average).mean())
        
        # top_indices_for_feature = np.argsort(scores_for_feature)[::-1]

        # # 创建条形图
        # plt.figure(figsize=(12, 6))
        # plt.bar([feature_name_list[i] for i in top_indices_for_feature], [scores_for_feature[i] for i in top_indices_for_feature])
        # plt.xlabel("Features")
        # plt.ylabel("Averaged Scores")
        # plt.title("Averaged Scores for Features (Sorted by Score)")
        # plt.xticks(rotation=90)
        # plt.tight_layout()
        # plt.savefig(f"result/fisher_score/all_sorted_fisher_scores_averaged_for_features.png")

        


if __name__ == "__main__":
    # with open('output.txt', 'w') as file:  # 将print内容保存到文件
    #     # 保存当前的标准输出
    #     original_stdout = sys.stdout
    #     # 将标准输出重定向到文件
    #     sys.stdout = file
    #     main()
    #     # 恢复原来的标准输出
    #     sys.stdout = original_stdout

    main() # 用于在终端输出print内容

import ast
import numpy as np
from matplotlib import pyplot as plt


def calculate_frr_far(confusion_matrix, positive_class):
    # 获取混淆矩阵的行数和列数
    num_classes = len(confusion_matrix)

    # 获取正样本对应的索引
    positive_index = positive_class

    # 计算 FRR
    false_negatives = sum(confusion_matrix[positive_index]) - confusion_matrix[positive_index][positive_index]
    actual_positives = sum(confusion_matrix[positive_index])
    frr = false_negatives / actual_positives

    # 计算 FAR
    false_positives = sum(confusion_matrix[i][positive_index] for i in range(num_classes) if i != positive_index)
    actual_negatives = sum(confusion_matrix[i][j] for i in range(num_classes) for j in range(num_classes) if
                           i != positive_index and j != positive_index)
    # print(false_positives, actual_negatives)
    far = false_positives / (actual_negatives + false_positives)

    return frr, far


def read_confusion_matrices_from_file(file_path, num_classes=17):
    with open(file_path, "r") as file:
        lines = file.readlines()

    confusion_allsize = []
    confusion_matrices = []  # 存储所有符合条件的混淆矩阵
    current_classifier_name = None
    current_confusion_matrix = None
    current_accuracy = None
    index = 0
    num_classes = num_classes

    for line in lines:
        if line.endswith("*** \n"):
            if confusion_matrices is not None:
                confusion_allsize.append(confusion_matrices.copy())  # 列表需要copy
        # print(line, index)
        if line.startswith("f1"):
            # 如果遇到分隔线，则说明前一个分类器的信息结束，将其保存到列表中
            if current_classifier_name is not None and current_confusion_matrix is not None and current_accuracy is not None:
                confusion_matrices.append((current_classifier_name, current_confusion_matrix))

        elif line.startswith("confusion:"):
            # print(index)
            # 开始提取新的分类器信息
            current_classifier_name = lines[index - 2].strip().replace("-", "")
            start_index = index + 1
            end_index = start_index + num_classes
            current_confusion_matrix = lines[start_index:end_index]
            confusion_matrix_str = ''.join(current_confusion_matrix)
            confusion_matrix_str = confusion_matrix_str.replace('[', '').replace(']', '').replace('\n', '')
            # print(confusion_matrix_str, len(confusion_matrix_str.replace(" ", "")))
            # 将字符串转换为 NumPy 数组
            # current_confusion_matrix = np.array(list(map(int, confusion_matrix_str.replace(" ", "")))).reshape(num_classes, num_classes)
            current_confusion_matrix = np.array([int(num) for num in confusion_matrix_str.split()]).reshape(num_classes,
                                                                                                            num_classes)
            print(current_confusion_matrix)
            # 删除多余的字符并将字符串转换为NumPy数组
            current_accuracy = 0

        index += 1

    return confusion_allsize

def main():
    # 示例文件路径
    dir = r"D:\pycharm\srt_vr_auth"
    file_path = dir + r"\src\1saccade4model20times.txt"
    num_classes = 17

    # 读取所有符合条件的混淆矩阵
    confusion_allsize = read_confusion_matrices_from_file(file_path, num_classes=num_classes)

    far_svm_all = []
    frr_svm_all = []
    far_knn_all = []
    frr_knn_all = []
    far_mlp_all = []
    frr_mlp_all = []
    far_rf_all = []
    frr_rf_all = []
    far_vote_all = []
    frr_vote_all = []
    # 打印所有混淆矩阵
    # print(confusion_allsize)
    for confusion_matrices in confusion_allsize:
        # print("num", confusion_matrices)
        far_svm = []
        frr_svm = []
        far_knn = []
        frr_knn = []
        far_mlp = []
        frr_mlp = []
        far_rf = []
        frr_rf = []
        far_vote = []
        frr_vote = []
        for classifier_name, confusion_matrix in confusion_matrices:
            # print("Classifier Name:", classifier_name)
            # print("Confusion Matrix:", confusion_matrix)
            for i in range(num_classes):
                # print(i)
                frr, far = calculate_frr_far(confusion_matrix, i)
                if classifier_name.startswith("svm"):
                    far_svm.append(far)
                    frr_svm.append(frr)
                elif classifier_name.startswith("knn"):
                    far_knn.append(far)
                    frr_knn.append(frr)
                elif classifier_name.startswith("mlp"):
                    far_mlp.append(far)
                    frr_mlp.append(frr)
                elif classifier_name.startswith("rf"):
                    far_rf.append(far)
                    frr_rf.append(frr)
                else:
                    far_vote.append(far)
                    frr_vote.append(frr)

        # Calculate mean and standard deviation for each list
        mean_far_svm = np.mean(far_svm)
        std_far_svm = np.std(far_svm) / np.sqrt(len(far_svm))

        mean_frr_svm = np.mean(frr_svm)
        std_frr_svm = np.std(frr_svm) / np.sqrt(len(frr_svm))

        mean_far_knn = np.mean(far_knn)
        std_far_knn = np.std(far_knn) / np.sqrt(len(far_svm))

        mean_frr_knn = np.mean(frr_knn)
        std_frr_knn = np.std(frr_knn) / np.sqrt(len(frr_svm))

        mean_far_mlp = np.mean(far_mlp)
        std_far_mlp = np.std(far_mlp) / np.sqrt(len(far_svm))

        mean_frr_mlp = np.mean(frr_mlp)
        std_frr_mlp = np.std(frr_mlp) / np.sqrt(len(frr_svm))

        mean_frr_rf = np.mean(frr_rf)
        std_frr_rf = np.std(frr_rf) / np.sqrt(len(frr_svm))

        mean_far_rf = np.mean(far_rf)
        std_far_rf = np.std(far_rf) / np.sqrt(len(frr_svm))

        mean_far_vote = np.mean(far_vote)
        std_far_vote = np.std(far_vote) / np.sqrt(len(far_vote))

        mean_frr_vote = np.mean(frr_vote)
        std_frr_vote = np.std(frr_vote) / np.sqrt(len(frr_vote))

        far_svm_all.append([mean_far_svm, std_far_svm])
        frr_svm_all.append([mean_frr_svm, std_frr_svm])

        far_knn_all.append([mean_far_knn, std_far_knn])
        frr_knn_all.append([mean_frr_knn, std_frr_knn])

        far_mlp_all.append([mean_far_mlp, std_far_mlp])
        frr_mlp_all.append([mean_frr_mlp, std_frr_mlp])

        far_rf_all.append([mean_far_rf, std_far_rf])
        frr_rf_all.append([mean_frr_rf, std_frr_rf])

        far_vote_all.append([mean_far_vote, std_far_vote])
        frr_vote_all.append([mean_frr_vote, std_frr_vote])

        print(far_vote_all)

    # 将数据转换为NumPy数组
    far_svm_all = np.array(far_svm_all)
    frr_svm_all = np.array(frr_svm_all)
    far_knn_all = np.array(far_knn_all)
    frr_knn_all = np.array(frr_knn_all)
    far_mlp_all = np.array(far_mlp_all)
    frr_mlp_all = np.array(frr_mlp_all)
    far_rf_all = np.array(far_rf_all)
    frr_rf_all = np.array(frr_rf_all)
    far_vote_all = np.array(far_vote_all)
    frr_vote_all = np.array(frr_vote_all)
    print(far_vote_all)
    print(frr_vote_all)

    # 生成 x 轴的数据，每个长度值对应三个点（SVM、KNN、MLP）
    x = range(1, len(far_svm_all))
    # print("len", x)
    #
    # # 计算纵轴的范围
    y_min = min(np.min(far_svm_all[1:, 0]), np.min(far_knn_all[1:, 0]), np.min(far_mlp_all[1:, 0]),
                np.min(far_rf_all[1:, 0])) * 100
    y_max = max(np.max(far_svm_all[1:, 0]), np.max(far_knn_all[1:, 0]), np.max(far_mlp_all[1:, 0]),
                np.max(far_rf_all[1:, 0])) * 100
    # y_max = max(np.max(far_vote_all[1:, 0]), np.max(frr_vote_all[1: 0])) * 100

    frr_means = frr_vote_all[1:, 0] * 100
    frr_std = frr_vote_all[1:, 1] * 100
    far_means = far_vote_all[1:, 0] * 100
    far_std = far_vote_all[1:, 1] * 100
    # plt.figure(figsize=(10, 6.6))
    # # 折线图
    # plt.errorbar(x, frr_means, yerr=frr_std, label='FRR', marker='o', linestyle='-', capsize=3)
    # plt.errorbar(x, far_means, yerr=far_std, label='FAR', marker='o', linestyle='-', capsize=3)
    #
    # # 添加标签、标题和图例
    # plt.xlabel('Number of Training Trials', fontsize=22)
    # plt.ylabel('FRR / FAR (%)', fontsize=22)
    # # plt.title('FRR and FAR with Error Bars')
    # plt.xticks(x, [f'{i + 2}' for i in range(len(frr_means))])
    # plt.legend(fontsize=22)
    # plt.grid(True)
    # # 设置X轴和Y轴上标的数的字体大小
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    # plt.yticks(np.arange(0, 3, 0.6))
    # plt.ylim(0, 2.8)
    # # 移除右侧和上侧的边框
    # plt.gca().spines['right'].set_visible(False)
    # plt.gca().spines['top'].set_visible(False)
    # plt.tight_layout(rect=[0, 0, 1, 0.95])
    # plt.show()

    # x = np.arange(1, len(frr_means) + 1)
    #
    # # Plot FRR in the first figure
    # plt.figure(figsize=(10, 6.6))
    # plt.errorbar(x, frr_means, yerr=frr_std, label='FRR', marker='o', linestyle='-', capsize=3)
    # plt.xlabel('Number of Training Trials', fontsize=22)
    # plt.ylabel('FRR (%)', fontsize=22)
    # plt.xticks(x, [f'{i + 2}' for i in range(len(frr_means))])
    # plt.legend(fontsize=22)
    # plt.grid(axis='y', linestyle='--', linewidth=1)  # Show only horizontal grid lines
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    # plt.yticks(np.arange(0, 3, 0.6))
    # plt.ylim(0, 2.8)
    # plt.gca().spines['right'].set_visible(False)
    # plt.gca().spines['top'].set_visible(False)
    # plt.tight_layout(rect=[0, 0, 1, 0.95])
    # plt.show()
    #
    # # Plot FAR in the second figure
    # plt.figure(figsize=(10, 6.6))
    # plt.errorbar(x, far_means, yerr=far_std, color='orange', label='FAR', marker='o', linestyle='-', capsize=3)
    # plt.xlabel('Number of Training Trials', fontsize=22)
    # plt.ylabel('FAR (%)', fontsize=22)
    # plt.xticks(x, [f'{i + 2}' for i in range(len(far_means))])
    # plt.legend(fontsize=22)
    # plt.grid(axis='y', linestyle='--', linewidth=1)  # Show only horizontal grid lines
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    # plt.yticks(np.arange(0, 0.123, 0.03))
    # plt.ylim(0, 0.123)
    # plt.gca().spines['right'].set_visible(False)
    # plt.gca().spines['top'].set_visible(False)
    # plt.tight_layout(rect=[0, 0, 1, 0.95])
    # plt.show()



    print("max", y_max)
    # 绘制 FAR 的折线图
    plt.figure(figsize=(10, 6.6))
    plt.errorbar(x, far_svm_all[1:, 0] * 100, yerr=far_svm_all[1:, 1] * 40, label='SVM', marker='o', capsize=3)
    plt.errorbar(x, far_knn_all[1:, 0] * 100, yerr=far_knn_all[1:, 1] * 40, label='KNN', marker='o', capsize=3)
    plt.errorbar(x, far_mlp_all[1:, 0] * 100, yerr=far_mlp_all[1:, 1] * 40, label='MLP', marker='o', capsize=3)
    plt.errorbar(x, far_rf_all[1:, 0] * 100, yerr=far_rf_all[1:, 1] * 40, label='RF', marker='o', capsize=3)

    # plt.title('False Acceptance Rate (FAR) for Different Classifiers')
    plt.xlabel('Number of Training Trials', fontsize=22)
    plt.ylabel('FAR (%)', fontsize=22)  # 修改纵轴标签
    # plt.legend(fontsize=18)

    plt.grid(axis='y', linestyle='--', linewidth=1)  # Show only horizontal grid lines
    plt.yticks(np.arange(0, y_max + 0.3, 0.5))  # 设置纵坐标刻度，从0到y_max，每隔1%
    plt.ylim(0, y_max + 0.3)  # 设置纵轴范围

    # 设置X轴和Y轴上标的数的字体大小
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # 移除右侧和上侧的边框
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # 保存图像
    # plt.savefig(dir + '/result/model_selection_1/FAR_plot.png')
    # 计算纵轴的范围
    y_min = min(np.min(frr_svm_all[1:, 0]), np.min(frr_knn_all[1:, 0]), np.min(frr_mlp_all[1:, 0]),
                np.min(frr_rf_all[1:, 0])) * 100
    y_max = max(np.max(frr_svm_all[1:, 0]), np.max(frr_knn_all[1:, 0]), np.max(frr_mlp_all[1:, 0]),
                np.max(frr_rf_all[1:, 0])) * 100

    # 绘制 FRR 的折线图
    plt.figure(figsize=(11.5, 6.6))
    plt.errorbar(x, frr_svm_all[1:, 0] * 100, yerr=frr_svm_all[1:, 1] * 40, fmt='-o', label='SVM', capsize=3)
    plt.errorbar(x, frr_knn_all[1:, 0] * 100, yerr=frr_knn_all[1:, 1] * 40, fmt='-o', label='KNN', capsize=3)
    plt.errorbar(x, frr_mlp_all[1:, 0] * 100, yerr=frr_mlp_all[1:, 1] * 40, fmt='-o', label='MLP', capsize=3)
    plt.errorbar(x, frr_rf_all[1:, 0] * 100, yerr=frr_rf_all[1:, 1] * 40, fmt='-o', label='RF', capsize=3)

    # plt.title('False Rejection Rate (FRR) for Different Classifiers')
    plt.xlabel('Number of Training Trials', fontsize=22)
    plt.ylabel('FRR (%)', fontsize=22)  # 修改纵轴标签
    plt.legend(fontsize=18)

    plt.grid(axis='y', linestyle='--', linewidth=1)  # Show only horizontal grid lines
    plt.yticks(np.arange(0, y_max + 3, 10))  # 设置纵坐标刻度，从0到y_max，每隔?%
    plt.ylim(0, y_max + 3)  # 设置纵轴范围

    plt.legend(fontsize=20, loc='center left', bbox_to_anchor=(1, 0.5))  # 设置loc参数和bbox_to_anchor参数
    # 设置X轴和Y轴上标的数的字体大小
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # 移除右侧和上侧的边框
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    plt.savefig(dir + '/result/model_selection_1/FRR_plot.png')

if __name__ == '__main__':
    main()
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
                confusion_allsize.append(confusion_matrices.copy()) #列表需要copy
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
            current_confusion_matrix = np.array([int(num) for num in confusion_matrix_str.split()]).reshape(num_classes, num_classes)
            print(current_confusion_matrix)
            # 删除多余的字符并将字符串转换为NumPy数组
            current_accuracy = 0

        index += 1

    return confusion_allsize

def main():
    # 示例文件路径
    file_path = r"D:\pycharm\srt_vr_auth\src\output2024-01-25-03-16-31-592802.txt"
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
    # 打印所有混淆矩阵
    print(confusion_allsize)
    for confusion_matrices in confusion_allsize:
        print("num", confusion_matrices)
        far_svm = []
        frr_svm = []
        far_knn = []
        frr_knn = []
        far_mlp = []
        frr_mlp = []
        far_rf = []
        frr_rf = []
        for classifier_name, confusion_matrix in confusion_matrices:
            print("Classifier Name:", classifier_name)
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

        far_svm_all.append([mean_far_svm, std_far_svm])
        frr_svm_all.append([mean_frr_svm, std_frr_svm])

        far_knn_all.append([mean_far_knn, std_far_knn])
        frr_knn_all.append([mean_frr_knn, std_frr_knn])

        far_mlp_all.append([mean_far_mlp, std_far_mlp])
        frr_mlp_all.append([mean_frr_mlp, std_frr_mlp])

        far_rf_all.append([mean_far_rf, std_far_rf])
        frr_rf_all.append([mean_frr_rf, std_frr_rf])

        print(far_svm_all)

    # 将数据转换为NumPy数组
    far_svm_all = np.array(far_svm_all)
    frr_svm_all = np.array(frr_svm_all)
    far_knn_all = np.array(far_knn_all)
    frr_knn_all = np.array(frr_knn_all)
    far_mlp_all = np.array(far_mlp_all)
    frr_mlp_all = np.array(frr_mlp_all)
    far_rf_all = np.array(far_rf_all)
    frr_rf_all = np.array(frr_rf_all)
    print(far_svm_all)


    # 生成 x 轴的数据，每个长度值对应三个点（SVM、KNN、MLP）
    x = range(len(far_svm_all))
    print("len", x)

    # 绘制 FAR 的折线图
    plt.figure(figsize=(10, 6))
    plt.errorbar(x, far_svm_all[:, 0], yerr=far_svm_all[:, 1], label='SVM', marker='o')
    plt.errorbar(x, far_knn_all[:, 0], yerr=far_knn_all[:, 1], label='KNN', marker='o')
    plt.errorbar(x, far_mlp_all[:, 0], yerr=far_mlp_all[:, 1], label='MLP', marker='o')
    plt.errorbar(x, far_rf_all[:, 0], yerr=far_rf_all[:, 1], label='RF', marker='o')

    plt.title('False Acceptance Rate (FAR) for Different Classifiers')
    plt.xlabel('Data Point Index')
    plt.ylabel('FAR')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 绘制 FRR 的折线图
    plt.figure(figsize=(10, 6))
    plt.errorbar(x, frr_svm_all[:, 0], yerr=frr_svm_all[:, 1], label='SVM', marker='o')
    plt.errorbar(x, frr_knn_all[:, 0], yerr=frr_knn_all[:, 1], label='KNN', marker='o')
    plt.errorbar(x, frr_mlp_all[:, 0], yerr=frr_mlp_all[:, 1], label='MLP', marker='o')
    plt.errorbar(x, frr_rf_all[:, 0], yerr=frr_rf_all[:, 1], label='RF', marker='o')

    plt.title('False Rejection Rate (FRR) for Different Classifiers')
    plt.xlabel('Data Point Index')
    plt.ylabel('FRR')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
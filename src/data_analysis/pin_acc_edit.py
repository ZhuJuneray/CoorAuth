import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import scipy.stats as stats

def find_matches(main_string, pattern):
    matches = re.finditer(f'(?=({pattern}))', main_string)
    return [match.group(1) for match in matches]


def longest_substring_matching(main_str, sub_str):
    main_len = len(main_str)
    sub_len = len(sub_str)

    # 构建一个二维数组来保存子问题的解
    dp = [[0] * (sub_len + 1) for _ in range(main_len + 1)]

    max_length = 0  # 最长匹配子串的长度
    end_index = 0  # 最长匹配子串在主串中的结束位置

    for i in range(1, main_len + 1):
        for j in range(1, sub_len + 1):
            if main_str[i - 1] == sub_str[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1

                # 更新最长匹配子串的信息
                if dp[i][j] > max_length:
                    max_length = dp[i][j]
                    end_index = i - 1
            else:
                dp[i][j] = 0

    if max_length == 0:
        return "没有找到匹配的子串"
    else:
        return main_str[end_index - max_length + 1: end_index + 1]


def longest_subsequence_ordered(main_str, sub_str):
    m, n = len(main_str), len(sub_str)

    # dp[i][j]表示主串前i个字符和子串前j个字符的最长公共子序列长度
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # 填充动态规划表
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if main_str[i - 1] == sub_str[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # 根据动态规划表构造最长子序列
    i, j = m, n
    longest_subseq = []
    while i > 0 and j > 0:
        if main_str[i - 1] == sub_str[j - 1]:
            longest_subseq.append(main_str[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return ''.join(reversed(longest_subseq))


def common_prefix_length(str1, str2):
    # 取两个字符串的前 n 位
    n = min(len(str1), len(str2))
    prefix_str1 = str1[:n]
    prefix_str2 = str2[:n]

    # 计算相同的字符数
    common_count = sum(1 for c1, c2 in zip(prefix_str1, prefix_str2) if c1 == c2)

    return common_count


# 指定文件夹路径
folder_path = os.path.join(os.getcwd(), 'data/VRAuthStudy1-1229/P9')

# 指定正则表达式模式
pattern = r'^PINEntry_study1-\d{1,2}-\d{4}-(\d+)-(\d+)-\d+.txt$'

# 创建字典，用于存储文件内容
file_dict = {}

# 遍历文件夹中的文件
for root, dirs, files in os.walk(folder_path):
    for file in files:
        # 构造文件的完整路径
        file_path = os.path.join(root, file)

        # 使用正则表达式匹配文件名
        match = re.match(pattern, file)

        if match:
            # 提取匹配的信息
            third_digit = int(match.group(1))
            second_digit = int(match.group(2))

            # 构建字典键
            key = (third_digit, second_digit)

            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 将文件内容添加到字典中
            if key not in file_dict:
                file_dict[key] = []
            file_dict[key].append(content)

print(file_dict)

# 正确的pin
# pin1 = '210367458'
# pin2 = '876452103'
# pin3 = '036785241'
# pin4 = '678501243'
pin1 = "67852"
pin2 = "6785"
pin3 = "6784"
pin4 = "6745"
pin5 = "6425"
pin6 = "04852"
pin7 = "63012"
pin8 = "84012"
pin9 = "67840"
pin10 = "3012"
pin11 = "67412"
pin12 = "7412"
pin13 = "6784012"
pin14 = "64258"
pin15 = "730125"
pin16 = "6785210"
pin17 = "6421"
pin18 = "48523"

find_dict = {1: pin1, 2: pin2, 3: pin3, 4: pin4, 5: pin5, 6: pin6, 7: pin7, 8: pin8, 9: pin9,
             10: pin10, 11: pin11, 12: pin12, 13: pin13, 14: pin14, 15: pin15, 16: pin16, 17: pin17, 18: pin18}

# print(find_dict)

all_correct_dict = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
ordered_correct_dict = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}

# 打印字典内容
for key, contents in file_dict.items():
    for content in contents:
        ordered_input = content.replace('-', '')
        main_s = find_dict[key[1]]
        # 按照最长匹配子串
        longest_match_ordered = common_prefix_length(main_s, ordered_input)
        num = longest_match_ordered
        ordered_correct_dict[key[0]].append(num / len(main_s))
        # 全部输入对才算对
        if main_s == ordered_input:
            num = 1
        else:
            num = 0
        all_correct_dict[key[0]].append(num)

print(len(all_correct_dict[2]))

#  存储std
all_sd_series=pd.Series(index=[1, 2, 3, 4, 5, 6])
# maybe_all_sd=pd.Series(index=[1,2,3,4,5,6])
ordered_sd_series=pd.Series(index=[1, 2, 3, 4, 5, 6])
# 计算 all_correct_dict 中每个键对应的值的平均值
for key, item in all_correct_dict.items():
    if key == 1 or key == 6:
        continue
    else:
        print(f"size: {key}, accuracy_ordered: {sum(item) / len(item)}")
        all_sd_series[key] = np.std(item) / np.sqrt(len(item))
        # all_sd_series[key]=math.sqrt(sum((item-mean_temp)*(item-mean_temp))/len(item))
# print(ordered_correct_dict)
# print(f'working?')
# print(all_sd_series)
# print(maybe_all_sd)
# 计算 ordered_correct_dict 中每个键对应的值的平均值
for key, item in ordered_correct_dict.items():
    if key == 1 or key == 6:
        continue
    else:
        print(f"size: {key}, accuracy_ordered: {sum(item) / len(item)}")
        ordered_sd_series[key] = np.std(item) / np.sqrt(len(item))
# 颜色方案
# 蓝色
blue_colors = plt.cm.Blues(np.linspace(0.25, 0.80, len(all_correct_dict)))
# 绿色
green_colors = plt.cm.Greens(np.linspace(0.25, 0.85, len(ordered_correct_dict)))
# Plot for all_correct_dict
sizes_all = [key for key in all_correct_dict if key not in [1, 6]]
accuracies_all = [sum(all_correct_dict[key]) / len(all_correct_dict[key]) for key in sizes_all]


# Plot for ordered_correct_dict
sizes_ordered = [key for key in ordered_correct_dict if key not in [1, 6]]
accuracies_ordered = [sum(ordered_correct_dict[key]) / len(ordered_correct_dict[key]) for key in sizes_ordered]

# Create a subplot with two columns
plt.figure(figsize=(8, 6))

# 定义柱状图宽度
bar_width = 0.5
# # Plot for all_correct_dict on the left
# bars_all = plt.bar(sizes_all, accuracies_all, color=blue_colors[4], width=0.5)
# # for i,key in enumerate(sizes_all):
# #     x_positions = np.arange(len(sizes_all))+i * bar_width
# #     plt.bar(x_positions,accuracies_all[i],label=key,width=bar_width,color=blue_colors[i])
# plt.set_xlabel('Size')
# plt.set_ylabel('Accuracy (All Correct Dict)')
# plt.set_title('Accuracy based on Size (All Correct Dict)')
#
# # Add labels at the top of each bar for all_correct_dict
# for bar in bars_all:
#     height = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.4f}', ha='center', va='bottom')

# mything[all]
all_color_by_height=[0, 2, 1, 3]
plt.xlabel('Size')
plt.ylabel('Accuracy (All Correct Dict)')
plt.title('Accuracy based on Size (All Correct Dict)')
# 这里是为了初始化一下x positions
x_positions=[1, 1.1, 1.2, 1.3]

for i in range(len(sizes_all)):
    x_positions[i] =(i) * bar_width
plt.bar(x_positions, accuracies_all, label=sizes_ordered, width=bar_width, color=blue_colors[all_color_by_height])  # 使用颜色列表中的颜色


plt.errorbar(x_positions, accuracies_all, yerr=all_sd_series[i+2], fmt=' ', capsize=5, label=f'{sizes_all[i]}' + '_std',ecolor='gray')
# cankao
# for key, item in all_correct_dict.items():
#     if key == 1 or key == 6:
#         continue
#     else:
#         print(f"size: {key}, accuracy_ordered: {sum(item) / len(item)}")
#
#         all_sd_series[key]=np.std(item)
# cankao

# mything[ordered]
order_color_by_index = [0, 3, 1, 2]

for i in range(len(sizes_ordered)):
    x_positions[i] = (i) * bar_width+5
plt.bar(x_positions, accuracies_ordered, label=sizes_ordered, width=bar_width, color=green_colors[order_color_by_index])  # 使用颜色列表中的颜色

# 误差线
plt.errorbar(x_positions, accuracies_ordered, yerr=ordered_sd_series[i+2], fmt=' ', capsize=5, label=f'{sizes_all[i]}' + '_std',ecolor='gray')

plt.show()

# 显著性绘图
# all的配对检验
for i in range(len(sizes_all)):
    x_positions[i] = (i) * bar_width

# 计算最大长度
max_all = 0
for key,values in all_correct_dict.items():
    if len(values)>max_all:
        max_all=len(values)

for k in range(2,5):
    for j in range(k+1,6):
        # print("bar",accuracies_all[i])
        # for key,item in all_correct_dict.items():
        #     #     if key == 1 or key == 6:
        #     #         continue
        #     #     else:
        #     #         print(f"size: {key}, accuracy_ordered: {sum(item) / len(item)}")
        #     #
        #     #         all_sd_series[key]=np.std(item)
        S_i = all_correct_dict[k]
        S_j = all_correct_dict[j]
        # 把长度补充到一样长
        max_length = max_all
        while(len(S_i)<max_all):
            # print(f'len now:{len(S_i)}')
            S_i.append(accuracies_all[k-2]) # 均值从第0存到第3，需要-2修正
        while(len(S_j)<max_all):
            # print(f'len now:{len(S_i)}')
            S_j.append(accuracies_all[j-2]) # 均值从第0存到第3，需要-2修正
        # print(f'final_ arr:{(S_i)}')
        # print(f'max_len?:{(max_length)}')
        # print("s working?",f'len:{k}___{j}',len(S_i),len(S_j))
        # 这里的代码得改
        t_stat, p_value = stats.ttest_rel(S_i,S_j)
        # print(t_stat,p_value,f'发现S{k},S{j}')
        # 检查p是否显著
        extraordinary = 0.05
        if p_value < extraordinary:
            print(f'发现准确率结果在size：{sizes_all[k-2]}———{sizes_all[j-2]}之间存在显著差异')
        if p_value < extraordinary:
            high_S_i = accuracies_all[k-2]  # 均值从第0存到第3，需要-2修正
            high_S_j = accuracies_all[j-2]  # 均值从第0存到第3，需要-2修正
            x1 = x_positions[k-2]   # 位置 从第0存到第3，需要-2修正
            x2 = x_positions[j-2]   # 位置 从第0存到第3，需要-2修正
            y = max(high_S_j,high_S_i)  + (k+2*j)*0.03-0.08

            x_points = [x1,x2]
            print(f'横坐标{k}{j}位置{x1}{x2}')
            y_points = [y, y]
            print(f'纵坐标位置')
            plt.plot(x_points,y_points,markersize = 5,linestyle= '-',color='black')
            # 在线段居中位置增加 '**'(纯文本) 或者 ‘*’（纯文本）
            plt.text((x1 + x2) / 2, y - 0.005, '**' if p_value < 0.01 else '*', ha='center', va='bottom',
                     color='black', fontsize=10, weight='bold')
            # plt.text((x1 + x2) / 2, y - 0.04, f'{k}__{j}', ha='center', va='bottom',
            #          color='black', fontsize=8, weight='bold')

            # 【随时撤销：引入细虚线】
            # 计算虚线的起点和终点
            y1_i = high_S_i
            y1_j = high_S_j
            y2_i = y
            y2_j = y
            x1_i = x2 = x_positions[k-2]
            x2_j = x_positions[j-2]

            # 绘制虚线
            # plt.plot([x1_i, x2], [y1_i, y2_i], linestyle='--', color='gray')
            # plt.plot([x2_j, x2_j], [y1_j, y2_j], linestyle='--', color='gray')

            plt.plot([x1_i, x2], [y2_i - 0.01, y2_i], linestyle='--', color='black')
            plt.plot([x2_j, x2_j], [y2_j - 0.01, y2_j], linestyle='--', color='black')
# 显著性绘图
# ordered
for i in range(len(sizes_ordered)):
    x_positions[i] = 5+(i) * bar_width
# 计算最大长度
max_all = 0
for key,values in ordered_correct_dict.items():
    if len(values)>max_all:
        max_all=len(values)

for k in range(2,5):
    for j in range(k+1,6):

        S_i = ordered_correct_dict[k]
        S_j = ordered_correct_dict[j]
        # 把长度补充到一样长
        max_length = max_all
        while(len(S_i)<max_all):
            # print(f'len now:{len(S_i)}')
            S_i.append(accuracies_ordered[k-2]) # 均值从第0存到第3，需要-2修正
        while(len(S_j)<max_all):
            # print(f'len now:{len(S_i)}')
            S_j.append(accuracies_ordered[j-2]) # 均值从第0存到第3，需要-2修正

        t_stat, p_value = stats.ttest_rel(S_i,S_j)
        # print(t_stat,p_value,f'发现S{k},S{j}')
        # 检查p是否显著
        extraordinary = 0.05
        if p_value < extraordinary:
            print(f'发现准确率结果在size：{sizes_ordered[k-2]}———{sizes_ordered[j-2]}之间存在显著差异')
        if p_value < extraordinary:
            high_S_i = accuracies_ordered[k-2]  # 均值从第0存到第3，需要-2修正
            high_S_j = accuracies_ordered[j-2]  # 均值从第0存到第3，需要-2修正
            x1 = x_positions[k-2]   # 位置 从第0存到第3，需要-2修正
            x2 = x_positions[j-2]   # 位置 从第0存到第3，需要-2修正
            y = max(high_S_j,high_S_i) + (3*k*0.04+2*j*0.03)*0.7-0.2

            # 修正最后一个柱子
            if k == 4:
                fix_k = 2
                y = max(high_S_j,high_S_i) + (3*3*0.04+2*j*0.03)*0.7-0.2

            x_points = [x1,x2]
            print(f'横坐标{k}{j}位置{x1}{x2}')
            y_points = [y, y]
            print(f'纵坐标位置')
            plt.plot(x_points,y_points,markersize=5,linestyle='-',color='black')
            # 在线段居中位置增加 '**'(纯文本) 或者 ‘*’（纯文本）
            plt.text((x1 + x2) / 2, y - 0.005, '**' if p_value < 0.01 else '*', ha='center', va='bottom',
                     color='black', fontsize=10, weight='bold')
            # plt.text((x1 + x2) / 2, y - 0.04, f'{k}__{j}', ha='center', va='bottom',
            #          color='black', fontsize=8, weight='bold')

            # 【随时撤销：引入细虚线】
            # 计算虚线的起点和终点
            y1_i = high_S_i
            y1_j = high_S_j
            y2_i = y
            y2_j = y
            x1_i = x2 = x_positions[k-2]
            x2_j = x_positions[j-2]

            # 绘制虚线
            # plt.plot([x1_i, x2], [y1_i, y2_i], linestyle='--', color='gray')
            # plt.plot([x2_j, x2_j], [y1_j, y2_j], linestyle='--', color='gray')

            plt.plot([x1_i, x2], [y2_i - 0.01, y2_i], linestyle='--', color='black')
            plt.plot([x2_j, x2_j], [y2_j - 0.01, y2_j], linestyle='--', color='black')

# Adjust layout for better spacing
# print(f'whats wrong with xposition:{x_positions}')
# plt.tight_layout()

# # 修改一下，把两个std只显示一个
# lines, labels = plt.gca().get_legend_handles_labels()
# selected_lines = [lines[i] if (i != 8) | (i!=9) else 'std' for i in [0,1,2,3,4,5,6,7,8,9]]
# selected_labels = [labels[i] if (i != 8) | (i!=9) else 'std' for i in [0,1,2,3,4,5,6,7,8,9]]

# plt.legend(selected_lines,selected_labels, bbox_to_anchor=(1.00, 0.5), loc='center left')
# Show the plot
plt.show()
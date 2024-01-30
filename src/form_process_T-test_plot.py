import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

def get_posthoc_results():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.stats as stats

    import csv, os, re
    import pandas as pd
    import statsmodels.api as sm
    from statsmodels.stats.anova import AnovaRM
    import pandas as pd
    import pingouin as pg

    # Define the path to your input CSV file and output CSV file
    input_csv_file = os.path.join(os.getcwd(), 'data/VRAuth1/S1form.csv')

    df = pd.read_csv(input_csv_file)

    # Prepare to store results
    rmanova_results = {}
    posthoc_results = {}

    # Performing Repeated Measures ANOVA for each question (Q1 to Q7)
    for question in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7']:
        df_subset = df[['P', 'S', question]].copy()
        df_subset = df_subset.rename(columns={question: 'Score'})
        # rmanova = AnovaRM(df_subset, 'Score', 'P', within=['S']).fit()
        # rmanova_results[question] = rmanova.summary()

        posthoc = pg.pairwise_ttests(data=df_subset, dv='Score', within='S', subject='P', padjust='bonferroni')
        posthoc_results[question] = posthoc

    # # Prepare an empty dictionary to store the results
    # results = {}
    # # Loop through each question (Q) and each S value (S)
    # for question in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7']:
    #     for s_value in [1,2,3,4]:
    #         subset = df[(df['S'] == s_value)][question]
    #         # Calculate the mean (M) and standard error (SE) for the filtered subset
    #         mean = subset.mean()
    #         std_error = subset.sem()

    #         # Store the results in the dictionary
    #         results[(question, s_value)] = {'M': mean, 'SE': std_error}

    # # Print the results
    # for key, value in results.items():
    #     print(f'Question: {key[0]}, S: {key[1]}, M: {value["M"]}, SE: {value["SE"]}')
    return  posthoc_results

posthoc_results = get_posthoc_results()
# 读取CSV文件
# C:\Users\Lenovo\Desktop
# file_path = r"C:\Users\Lenovo\Desktop\data\VRAuth1\S1form.xls"  # 替换为你的CSV文件路径
file_path = os.path.join(os.getcwd(), 'data/VRAuth1/S1form_to_clean.csv')  # 替换为你的CSV文件路径
df = pd.read_csv(file_path)  # 或者使用其他编码方式，如 encoding='gbk'

# 将DataFrame转换为字典列表
list_of_dicts = df.to_dict(orient='records')

# 创建字典来保存每个 Sx 对应问题的总和和计数
sum_scores_1 = {'Sx': "S1", 'Q1': [], 'Q2': [], 'Q3': [], 'Q4': [], 'Q5': [], 'Q6': [], 'Q7': []}
sum_scores_2 = {'Sx': "S2", 'Q1': [], 'Q2': [], 'Q3': [], 'Q4': [], 'Q5': [], 'Q6': [], 'Q7': []}
sum_scores_3 = {'Sx': "S3", 'Q1': [], 'Q2': [], 'Q3': [], 'Q4': [], 'Q5': [], 'Q6': [], 'Q7': []}
sum_scores_4 = {'Sx': "S4", 'Q1': [], 'Q2': [], 'Q3': [], 'Q4': [], 'Q5': [], 'Q6': [], 'Q7': []}
sum_scores_5 = {'Sx': "S5", 'Q1': [], 'Q2': [], 'Q3': [], 'Q4': [], 'Q5': [], 'Q6': [], 'Q7': []}
sum_scores_6 = {'Sx': "S6", 'Q1': [], 'Q2': [], 'Q3': [], 'Q4': [], 'Q5': [], 'Q6': [], 'Q7': []}

sum_scores_list = [sum_scores_1, sum_scores_2, sum_scores_3, sum_scores_4, sum_scores_5, sum_scores_6]

# 遍历字典列表
for row_dict in list_of_dicts:
    # 获取Sx值
    Sx = row_dict["Q1_A1_open"].split('-')[-1]

    # 遍历每个 Sx 对应的字典列表，找到匹配的字典
    for sum_scores in sum_scores_list:
        if sum_scores['Sx'] == Sx:
            # 遍历字典的键值对
            for key, value in row_dict.items():
                # 如果是问题的列（Q2, Q3, Q4, Q5），则将值加到对应的总和中
                if key.startswith('Q') and key.endswith('A1'):
                    question = key.split('_')[0]  # 获取问题的标识（Q2, Q3, Q4, Q5）
                    sum_scores[question[0] + str(int(question[1]) - 1)].append(8 - value)

# 将字典列表转换为DataFrame
df = pd.DataFrame(sum_scores_list)

print("df", df)

# 提取问题的列
questions_columns = df.columns[df.columns.str.startswith('Q')]

# 创建一个新的DataFrame，用于存储每个问题的平均值
result_df = pd.DataFrame(columns=['Question'] + [f"S{i}" for i in range(1, 7)])
# new data frame 储存标准差
sd_df = pd.DataFrame(columns=['Question'] + [f"S{i}" for i in range(1, 7)])
# 遍历每个问题列
for col in questions_columns:
    question_data = {'Question': col}
    sd_data = {'Question': col}
    # 遍历每个S列，计算平均值
    for i in range(1, 7):
        S_col = f"S{i}"
        question_data[S_col] = df[df['Sx'] == f"S{i}"][col].explode().astype(float).mean()
        n = len(df[df['Sx'] == f"S{i}"][col].explode().astype(float))
        sd_data[S_col] = df[df['Sx'] == f"S{i}"][col].explode().astype(float).std()/ np.sqrt(n)

    # 将结果添加到新的DataFrame中
    result_df = result_df._append(question_data, ignore_index=True)
    # 添加标准差
    sd_df = sd_df._append(sd_data, ignore_index=True)

print("result_df", result_df)
# 【不改变区域】
# 设置图形大小
plt.figure(figsize=(15, 6))

# 定义柱状图的宽度
bar_width = 0.15

# 创建一个颜色列表，用于存储渐进的蓝色
colors = plt.cm.Blues(np.linspace(0.15, 0.85, len(result_df.columns[1:])))
print(f"len(result_df.columns[1:]): {len(result_df.columns[1:])}")
# 【不改变区域】
# 遍历每个问题列，绘制柱状图
for i, col in enumerate(result_df.columns[1:]):
    x_positions = np.arange(len(result_df['Question'])) + i * bar_width
    plt.bar(x_positions, result_df[col], label=col, width=bar_width, color=colors[i])  # 使用颜色列表中的颜色

    std_error = sd_df[col]
    plt.errorbar(x_positions, result_df[col], yerr=std_error, fmt=' ', capsize=5, label=col + '_std',ecolor='gray')

# 画一道绿色的水平线
# 画一道淡绿色的水平线
for y_coord in range(1, 8):
    plt.axhline(y=y_coord, color=(0.8, 0.8, 0.8, 0.9), linestyle='--', linewidth=1)  # (0, 1, 0, 0.3)代表淡绿色


# 【恢复区】
# # 原始遍历绘图，随时恢复
# # 遍历每个问题列，绘制柱状图
# for i, col in enumerate(result_df.columns[1:]):
#     x_positions = np.arange(len(result_df['Question'])) + i * bar_width
#     plt.bar(x_positions, result_df[col], label=col, width=bar_width)

# 调整x轴标签的位置
plt.xticks(np.arange(len(result_df['Question'])) + (len(result_df.columns[1:]) - 1) * bar_width / 2, result_df['Question'])

# 显著性检测绘图
for i, col in enumerate(df.columns[1:]): # 从Q2-Q8
    x_positions = np.arange(len(result_df['Question'])) # 基础的position
# 对每一对S值执行t-配对检验
    for k in range(1, 4):
        for j in range(k + 1, 5):
            S_i = df[df['Sx'] == f"S{k}"][col].explode().astype(float)
            S_j = df[df['Sx'] == f"S{j}"][col].explode().astype(float)
            p_value = (posthoc_results[col][(posthoc_results[col]['A'] == k) & (posthoc_results[col]['B'] == j)]['p-corr'].values)[0]
            print(f"{col} S_k, S_{k}", f"S_j, S_{j}", f"p_value: {p_value}")
            # print("S_k", S_i, S_j)
            # t_stat, p_value = stats.ttest_rel(S_i, S_j)
            # print(t_stat, p_value, f"发现S{k}, S{j}")
            # 检查p值是否显著（可以根据需要调整显著水平）
            extraordinary = 0.05
            if p_value < extraordinary:
                print(f"发现{col}在S{k}和S{j}之间存在显著差异（p值: {p_value}）")
            if p_value < extraordinary:
                print(f"resulu_df: {result_df}")
                high_S_i = result_df.loc[result_df['Question'] == col]["S" + str(k)].values[0]
                high_S_j = result_df.loc[result_df['Question'] == col]["S" + str(j)].values[0]
                x1 = x_positions[int(str(col)[-1]) - 2] + k * bar_width
                x2 = x_positions[int(str(col)[-1]) - 2] + j * bar_width
                y = max(high_S_j, high_S_i) - 1.5 + (k + j + 2) * 0.3
                x_points = [x1, x2]
                # 这里尝试把Q8往上移动
                if y <= max(high_S_j, high_S_i)+0.03:
                    y += 0.6


                # if (y>=max(high_S_j, high_S_i)+2)&(p_value>=0.01):
                #     y-=0.5

                y_points = [y, y]
                # 在柱状图上添加标记（*表示显著，**表示极显著）
                # # 添加线段
                # 【原始绘图】
                # plt.plot(x_points, y_points, marker='.' if p_value < 0.01 else "*", markersize=10, linestyle='-', color='black')

                plt.plot(x_points, y_points,  markersize=10, linestyle='-', color='black')

                # 在线段居中位置增加 '**'(纯文本) 或者 ‘*’（纯文本）
                plt.text((x1 + x2) / 2, y - 0.08, '**' if p_value < 0.01 else '*', ha='center', va='bottom', color='black', fontsize=12, weight='bold')
                # 【随时撤销：引入细虚线】

                # [随时取消]
                # 计算虚线的起点和终点
                y1_i = high_S_i
                y1_j = high_S_j
                y2_i = y
                y2_j = y
                x1_i = x2 = x_positions[int(str(col)[-1]) - 2] + (k - 1) * bar_width
                x2_j = x_positions[int(str(col)[-1]) - 2] + (j - 1) * bar_width

                # 绘制虚线
                # plt.plot([x1_i, x2], [y1_i, y2_i], linestyle='--', color='gray')
                # plt.plot([x2_j, x2_j], [y1_j, y2_j], linestyle='--', color='gray')

                plt.plot([x1_i, x2], [y2_i-0.08, y2_i], linestyle='--', color='black')
                plt.plot([x2_j, x2_j], [y2_j-0.08, y2_j], linestyle='--', color='black')



# 添加图例和标签
# plt.legend()
# plt.legend(bbox_to_anchor=(1.005, 0.5), loc='center left')
# 获取当前图表的所有线条和标签
lines, labels = plt.gca().get_legend_handles_labels()

# 选择你想要在图例中显示的线条和标签
# 假设你只想显示S2,S3,S4,S5,std的图例
selected_lines = [lines[i] for i in [2, 4, 6, 8, 1]]
# selected_labels = [labels[i] for i in [ 2,  4, 6,8,1]]
selected_labels = [labels[i] if i != 1 else 'std' for i in [2, 4, 6, 8, 1]]


# 创建一个自定义的图例
plt.legend(selected_lines, selected_labels, bbox_to_anchor=(1.005, 0.5), loc='center left')

plt.title('x:Q2-Q5  y:S1-S6 mean')
plt.xlabel('Q')
plt.ylabel('Mean Score')

# # 显示图形
# plt.show()

# Define and create the plot folder
plot_folder = os.path.join("result/", "study1")
if not os.path.exists(plot_folder):
    os.makedirs(plot_folder)

# Define the plot filename
plot_filename = f"form_by_zs.png"
plt.savefig(os.path.join(plot_folder, plot_filename))
plt.close(plt)
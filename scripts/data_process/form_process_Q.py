import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# 读取CSV文件
file_path = "E:\Desktop\data\VRAuth1\S1form.xls"  # 替换为你的CSV文件路径
df = pd.read_excel(file_path)  # 或者使用其他编码方式，如 encoding='gbk'

# 将DataFrame转换为字典列表
list_of_dicts = df.to_dict(orient='records')

# 创建字典来保存每个 Sx 对应问题的总和和计数
sum_scores_1 = {'Sx': "S1", 'Q2': [], 'Q3': [], 'Q4': [], 'Q5': []}
sum_scores_2 = {'Sx': "S2", 'Q2': [], 'Q3': [], 'Q4': [], 'Q5': []}
sum_scores_3 = {'Sx': "S3", 'Q2': [], 'Q3': [], 'Q4': [], 'Q5': []}
sum_scores_4 = {'Sx': "S4", 'Q2': [], 'Q3': [], 'Q4': [], 'Q5': []}
sum_scores_5 = {'Sx': "S5", 'Q2': [], 'Q3': [], 'Q4': [], 'Q5': []}
sum_scores_6 = {'Sx': "S6", 'Q2': [], 'Q3': [], 'Q4': [], 'Q5': []}

sum_scores_list = [sum_scores_1, sum_scores_2, sum_scores_3, sum_scores_4, sum_scores_5, sum_scores_6]

# 遍历字典列表
for row_dict in list_of_dicts:
    # 获取Sx值
    Sx = row_dict["Q1_被试编码（Px-Sx）"].split('-')[-1]

    # 遍历每个 Sx 对应的字典列表，找到匹配的字典
    for sum_scores in sum_scores_list:
        if sum_scores['Sx'] == Sx:
            # 遍历字典的键值对
            for key, value in row_dict.items():
                # 如果是问题的列（Q2, Q3, Q4, Q5），则将值加到对应的总和中
                if key.startswith('Q') and key.endswith('选项1'):
                    question = key.split('_')[0]  # 获取问题的标识（Q2, Q3, Q4, Q5）
                    sum_scores[question].append(value)

# 将字典列表转换为DataFrame
df = pd.DataFrame(sum_scores_list)

# 提取问题的列
questions_columns = df.columns[df.columns.str.startswith('Q')]

# 创建一个新的DataFrame，用于存储每个问题的平均值
result_df = pd.DataFrame(columns=['Question'] + [f"S{i}" for i in range(1, 7)])

# 遍历每个问题列
for col in questions_columns:
    question_data = {'Question': col}
    # print(col)

    # 遍历每个S列，计算平均值
    for i in range(1, 7):
        S_col = f"S{i}"
        question_data[S_col] = df[df['Sx'] == f"S{i}"][col].explode().astype(float).mean()

    # 将结果添加到新的DataFrame中
    result_df = result_df.append(question_data, ignore_index=True)

print(result_df)

# 设置图形大小
plt.figure(figsize=(10, 6))

# 定义柱状图的宽度
bar_width = 0.15

# 遍历每个问题列，绘制柱状图
for i, col in enumerate(result_df.columns[1:]):
    x_positions = np.arange(len(result_df['Question'])) + i * bar_width
    plt.bar(x_positions, result_df[col], label=col, width=bar_width)

# 调整x轴标签的位置
plt.xticks(np.arange(len(result_df['Question'])) + (len(result_df.columns[1:]) - 1) * bar_width / 2, result_df['Question'])

# 添加图例和标签
plt.legend()
plt.title('x:Q2-Q5  y:S1-S6 mean')
plt.xlabel('Q')
plt.ylabel('Mean Score')

# 显示图形
plt.show()

for col in questions_columns:
    print(col)
    # 对每一对S值执行t-配对检验
    for i in range(1, 6):
        for j in range(i+1, 7):
            S_i = df[df['Sx'] == f"S{i}"][col].explode().astype(float)
            S_j = df[df['Sx'] == f"S{j}"][col].explode().astype(float)

            t_stat, p_value = stats.ttest_rel(S_i, S_j)

            # 检查p值是否显著（可以根据需要调整显著水平）
            extraordinary = 0.05
            if p_value < extraordinary:
                print(f"发现{col}在S{i}和S{j}之间存在显著差异（p值: {p_value}）")



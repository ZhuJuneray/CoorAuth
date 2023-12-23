import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# 打印每个 Sx 对应的字典
for sum_scores in sum_scores_list:
    print(sum_scores)

# 将字典列表转换为DataFrame
df = pd.DataFrame(sum_scores_list)

# 提取问题的列
questions_columns = df.columns[df.columns.str.startswith('Q')]

# 创建一个新的DataFrame，用于存储每个 S 对应问题的平均值
result_df = pd.DataFrame(columns=['Sx'] + [f"{col}_mean" for col in questions_columns])

# 遍历
for index, row in df.iterrows():
    Sx = row['Sx']
    row_data = {'Sx': Sx}

    # 计算每个问题的平均值和标准差
    for col in questions_columns:
        mean_value = pd.Series(row[col]).mean()
        std_value = pd.Series(row[col]).std()
        row_data[f"{col}_mean"] = mean_value
        row_data[f"{col}_std"] = std_value

    # 将结果添加到新的DataFrame中
    result_df = result_df.append(row_data, ignore_index=True)

# 打印结果
print(result_df)

# 设置图形大小
plt.figure(figsize=(10, 6))

# 定义每个问题列的偏移量
offset = np.array([-0.2, 0, 0.2, 0.4])

print(result_df['Sx'])

result_df['Sx'] = result_df['Sx'].astype(str)

# 遍历每个问题列，绘制柱状图
for i, col in enumerate(questions_columns):
    x_ticks = np.arange(len(result_df['Sx'])) + offset[i]
    plt.bar(x_ticks, result_df[f'{col}_mean'], yerr=result_df[f'{col}_std'], label=col, alpha=0.7, width=0.2)

# 设置 X 轴刻度
plt.xticks(np.arange(len(result_df['Sx'])), result_df['Sx'])

# 添加图例和标签
plt.legend()
plt.title('Average Scores and Standard Deviation by S')
plt.xlabel('Sx')
plt.ylabel('Score')

# 显示图形
plt.show()


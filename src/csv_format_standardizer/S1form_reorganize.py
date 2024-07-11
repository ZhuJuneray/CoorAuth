import os, re
import pandas as pd

# Define the path to your input CSV file and output CSV file
input_csv_file = os.path.join(os.getcwd(), 'data/VRAuth1/S1form_to_clean.csv')
output_csv_file = os.path.join(os.getcwd(), 'data/VRAuth1/S1form.csv')

df = pd.read_csv(input_csv_file)
# 读取第一列的数据
first_column = df.iloc[:, 0]
value_1 = [re.findall(r'P(\d+)', i.split('-')[0])[0] for i in first_column]
value_2 = [re.findall(r'S(\d+)', i.split('-')[1])[0] for i in first_column]

# 删除第一列
df = df.drop(df.columns[0], axis=1)

# 插入两个新列在前面
df.insert(0, 'P', value_1)
df.insert(1, 'S', value_2)

# 创建一个新的列头名称的字典
new_column_names = {'Q2_A1': 'Q1', 'Q3_A1': 'Q2', 'Q4_A1': 'Q3', 'Q5_A1': 'Q4', 'Q6_A1': 'Q5', 'Q7_A1': 'Q6', 'Q8_A1': 'Q7',}

# 使用rename方法重命名列头
df = df.rename(columns=new_column_names)

# 将效果好对应低分转换为高分
df[['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7']] = 8 - df[['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7']]
df[['S']] = df[['S']].astype(int) - 1

print(df)

df.to_csv(output_csv_file, index=False, header=True)

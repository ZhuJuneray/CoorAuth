import os
import pandas as pd

# 新表头格式
new_headers = ['H-Vector3X' , 'H-Vector3Y' , 'H-Vector3Z' , 'H-QuaternionX' , 'H-QuaternionY' , 'H-QuaternionZ' , 'H-QuaternionW']

# 设置存放CSV文件的目录
directory = os.getcwd()

# 遍历目录中的所有文件
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        file_path = os.path.join(directory, filename)

        # 读取CSV文件
        df = pd.read_csv(file_path)

        # 检查原始表头长度与新表头长度是否一致
        if len(df.columns) == len(new_headers):
            # 替换表头
            df.columns = new_headers

            # 保存更改
            df.to_csv(file_path, index=False)
        else:
            print(f"表头长度不匹配：{filename}")

print("所有CSV文件的表头已更新。")

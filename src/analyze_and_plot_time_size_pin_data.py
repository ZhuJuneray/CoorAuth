import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from data_preprocess import read_data_name_from_json
from collections import defaultdict
def analyze_and_plot_time_size_pin_data(filepath="src/data.json", size_list = [1, 2, 3, 4, 5, 6], pin_list = [1, 2, 3, 4]): #画每个size的用时
    # Initialize parameters
    
    rotdir = os.path.join(os.getcwd(), 'data/')
    studytype_users_dates = read_data_name_from_json(filepath)

    # Prepare a dictionary to store time data and read files to calculate average times
    times = defaultdict(lambda: defaultdict(list))
    for member in studytype_users_dates:
        studytype, user, date = member.split('-')
        for size in size_list:
            for pin in pin_list:
                prefix = f"Head_data_{studytype}-{user}-{date}-{size}-{pin}-"
                for file in os.listdir(rotdir + f"data{date}/P{user}"):
                    if file.startswith(prefix) and file.endswith(".csv"):
                        file_path = os.path.join(rotdir + f"data{date}/P{user}", file)
                        data = pd.read_csv(file_path)
                        time = len(data) * 0.02 - 0.1
                        if time <= 20:  # Exclude unreasonable times
                            times[size][pin].append(time)

    # Calculate average times for each size and pin condition
    # 计算每种size和pin条件下的平均用时
    # 计算每种size和pin条件下的平均用时
    average_times = defaultdict(dict)
    for size, pins in times.items():
        for pin, time_list in pins.items():
            if time_list:  # Ensure we don't divide by zero
                average_times[size][pin] = sum(time_list) / len(time_list)

    # 输出结果
    for size, pins in average_times.items():
        for pin, avg_time in pins.items():
            print(f"Size {size}, Pin {pin}, Average Time: {avg_time}")



    # 假设 average_times 是前面计算出来的平均用时的嵌套字典

    # 转换数据结构以适应绘图需要
    data = {}
    for size in size_list:
        for pin in pin_list:
            data.setdefault(size, []).append(average_times.get(size, {}).get(pin, 0))

    # 绘制分组条形图
    size_indices = np.arange(len(size_list))  # size的x轴位置
    bar_width = 0.2  # 条形的宽度

    # 绘制每个pin的条形
    for i, pin in enumerate(pin_list):
        # 提取每个size下特定pin的平均用时
        pin_times = [data[size][i] for size in size_list]
        # 绘制条形图
        plt.bar(size_indices + i * bar_width, pin_times, width=bar_width, label=f'Pin {pin}')

    # 设置图表标题和标签
    plt.xlabel('Size')
    plt.ylabel('Average Time (s)')
    plt.title('Average Time for Different Sizes and Pins')
    plt.xticks(size_indices + bar_width * (len(pin_list)-1)/2, size_list)  # 设置x轴标签位置和标签名
    plt.legend(title="Pin")  # 添加图例

    # Create result folder if not exist
    folder_name = os.path.join(os.getcwd(), "result")
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Save the figure
    plt.savefig(os.path.join(folder_name, "average_time_for_different_sizes_and_pins.png"))
    plt.close()  # Close the plot to avoid displaying it when not needed
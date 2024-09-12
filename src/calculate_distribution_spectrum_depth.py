import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import gaussian_kde

# 读取Saccades的范围并计算补集（Fixation范围）
def get_fixation_ranges(saccades_ranges, total_length):
    saccades = [tuple(map(int, item.split('-'))) for item in saccades_ranges.split(';') if item]
    fixation_ranges = []
    current_start = 0
    for sac_start, sac_end in saccades:
        if current_start < sac_start:
            fixation_ranges.append((current_start, sac_start - 1))
        current_start = sac_end + 1
    if current_start < total_length:
        fixation_ranges.append((current_start, total_length - 1))
    return fixation_ranges

# 计算 THM
def calculate_thm(df):
    thms = []
    for index, row in df.iterrows():
        yaw = np.abs(np.degrees(np.arctan2(2 * (row['H-QuaternionW'] * row['H-QuaternionZ'] + row['H-QuaternionX'] * row['H-QuaternionY']),
                                           1 - 2 * (row['H-QuaternionY']**2 + row['H-QuaternionZ']**2))))
        pitch = np.abs(np.degrees(np.arcsin(2 * (row['H-QuaternionW'] * row['H-QuaternionY'] - row['H-QuaternionZ'] * row['H-QuaternionX']))))
        thm = np.arccos(1 - 2 * (np.sin(np.radians(yaw / 2))**2 + np.sin(np.radians(pitch / 2))**2))
        thms.append(np.degrees(np.abs(thm)))
    return thms

# 计算 Distribution Spectrum Depth (DSD)
def calculate_dsd(thms):
    kde = gaussian_kde(thms, bw_method=0.2)
    x_grid = np.linspace(0, 100, 1000)  # 生成 THM 值的网格
    density = kde(x_grid)

    # 计算峰的高度（最大密度值）
    peak_height = np.max(density)

    # 计算频谱的宽度，定义为包含 90% 密度的范围
    cumulative_density = np.cumsum(density) / np.sum(density)  # 计算累计密度
    lower_bound = x_grid[np.searchsorted(cumulative_density, 0.05)]  # 下边界（5%）
    upper_bound = x_grid[np.searchsorted(cumulative_density, 0.95)]  # 上边界（95%）
    spectrum_width = upper_bound - lower_bound

    # 计算 DSD
    dsd = spectrum_width / peak_height
    return dsd, spectrum_width, peak_height

# 定义数据目录，修改为父目录“data/VRAuth2”，处理每个子文件夹
parent_dir = "data/VRAuth2"  # 替换为你的根文件夹路径

thm_data = {}
participants = []
dsd_data = {}  # 保存 DSD 结果

# 遍历 parent_dir 目录下的所有子文件夹
for subdir in os.listdir(parent_dir):
    subdir_path = os.path.join(parent_dir, subdir)
    
    if os.path.isdir(subdir_path):  # 确保处理的是子文件夹
        for file in os.listdir(subdir_path):
            if file.startswith('Head_data_study') and file.endswith('.csv'):
                # 提取文件名中的信息
                parts = file.split('-')
                study = parts[0].split('_')[2]
                participant = int(parts[1])
                date = parts[2]
                size = int(parts[3])
                pin_type = int(parts[4])
                repeat_time = int(parts[5].split('.')[0])

                # 根据 study 号动态生成 PINEntry 和 Saccades 文件名
                pin_entry_file = os.path.join(subdir_path, f"PINEntry_{study}-{participant}-{date}-{size}-{pin_type}-{repeat_time}.txt")
                saccades_file = os.path.join(subdir_path, f"Saccades_{study}-{participant}-{date}-{size}-{pin_type}-{repeat_time}.txt")
                
                # 尝试读取 PINEntry 和 Saccades 文件
                try:
                    with open(pin_entry_file, 'r') as f:
                        pin_entry = f.read().strip().split('-')  # 获取 PINEntry 输入的点
                        pin_entry.pop(0)  # 删除第一个元素
                except FileNotFoundError:
                    print(f"File not found: {pin_entry_file}, skipping this entry.")
                    continue

                try:
                    with open(saccades_file, 'r') as f:
                        saccades_ranges = f.read().strip()  # 读取 Saccades 文件内容
                except FileNotFoundError:
                    print(f"File not found: {saccades_file}, skipping this entry.")
                    continue

                # 读取对应的 Head_data 文件
                head_data_file = os.path.join(subdir_path, file)
                head_data = pd.read_csv(head_data_file)

                # 获取 fixation 时间范围
                fixation_ranges = get_fixation_ranges(saccades_ranges, len(head_data))

                # 对于每一个 fixation range，计算 THM
                for fixation_range, pin in zip(fixation_ranges, pin_entry):
                    pin = int(pin)
                    fixation_data = head_data.iloc[fixation_range[0]:fixation_range[1] + 1]
                    thms = calculate_thm(fixation_data)
                    if participant not in thm_data:
                        thm_data[participant] = []
                    thm_data[participant].extend(thms)

# 计算每个参与者的 DSD 并打印结果
for participant, thms in thm_data.items():
    dsd, spectrum_width, peak_height = calculate_dsd(thms)
    dsd_data[participant] = (dsd, spectrum_width, peak_height)
    print(f'Participant {participant}: DSD={dsd:.2f}, Spectrum Width={spectrum_width:.2f}, Peak Height={peak_height:.2f}')

# 你可以选择将 DSD 数据进行进一步的分析或绘图

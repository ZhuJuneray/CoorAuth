import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import gaussian_kde

# (size, pin) 对应的 target distance 映射
target_distance_mapping = {
    (3, 0): 30,
    (3, 1): 22.2,
    (3, 2): 30,
    (3, 3): 22.2,
    (3, 4): 0,
    (3, 5): 22.2,
    (3, 6): 30,
    (3, 7): 22.2,
    (3, 8): 30,
}

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

# 计算 DSD (Distribution Spectrum Depth)
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
dsd_data = {}  # 存储每个参与者的 DSD
participants = []
dsd_threshold = 400  # DSD 的阈值

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
                    target_distance = target_distance_mapping.get((size, pin), None)
                    if target_distance is not None:
                        fixation_data = head_data.iloc[fixation_range[0]:fixation_range[1] + 1]
                        thms = calculate_thm(fixation_data)
                        if (target_distance, participant) not in thm_data:
                            thm_data[(target_distance, participant)] = []
                        thm_data[(target_distance, participant)].extend(thms)

# 创建频数分布图，根据 target_distance 分组
target_distance_groups = {}

# 按 target_distance 对数据进行分组
for (target_distance, participant), thms in thm_data.items():
    if target_distance not in target_distance_groups:
        target_distance_groups[target_distance] = {}
    target_distance_groups[target_distance][participant] = thms
    # 计算每个参与者的 DSD
    dsd, spectrum_width, peak_height = calculate_dsd(thms)
    dsd_data[participant] = dsd

# 颜色映射
colors = cm.rainbow(np.linspace(0, 1, len(thm_data)))

# 遍历每个 target_distance 分组并绘制单独的图
for target_distance, group_data in target_distance_groups.items():
    plt.figure(figsize=(10, 6))
    print(f'Plotting for target distance: {target_distance}°')

    # 绘制每个 participant 的 THM 频数分布（使用核密度估计）
    for i, (participant, thms) in enumerate(group_data.items()):
        # 过滤掉 P17 及其后的数据
        if participant >= 17:
            continue

        kde = gaussian_kde(thms, bw_method=0.2)  # 通过 bw_method 设置平滑度
        x_grid = np.linspace(0, 100, 1000)  # 生成 THM 值的网格

        # 根据 DSD 的阈值进行分类
        if dsd_data[participant] > dsd_threshold:
            # 如果 DSD 大于阈值，使用红色加粗线条
            plt.plot(x_grid, kde(x_grid), color=colors[i], linestyle='-', label=f'P{participant}')
        else:
            # 如果 DSD 小于或等于阈值，使用默认样式
            plt.plot(x_grid, kde(x_grid), color=colors[i], linestyle='-', label=f'P{participant}')

    # 图例、标签和标题
    plt.xlabel('THM (degree)')
    plt.ylabel('Normalized Density')
    plt.figtext(0.5, -0.1, f'THM Distribution for Target Distance {target_distance}°', ha="center")
    # plt.title(f'THM Distribution for Target Distance {target_distance}°')
    plt.xlim(1, 50)
    # 将图例设置为两列
    plt.legend(loc='best', title='Participants', ncol=2)
    plt.show()

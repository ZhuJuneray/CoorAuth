import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from cycler import cycler
import os

def replace_local_outliers(arr, window_size=5, threshold=1.5): #去除离群值
    """
    使用滑动窗口方法替换一维数组中的局部离群值。

    参数:
    arr: 一维数组，可以是列表或NumPy数组
    window_size: 滑动窗口的大小
    threshold: 离群值的阈值，基于局部IQR

    返回:
    替换局部离群值后的数组
    """
    arr = np.array(arr)
    half_window = window_size // 2
    n = len(arr)

    for i in range(n):
        # 定义窗口的开始和结束索引
        start = max(0, i - half_window)
        end = min(n, i + half_window)

        # 提取窗口内的数据
        window_data = arr[start:end]

        # 计算四分位数和IQR
        Q1 = np.percentile(window_data, 0.25)
        Q3 = np.percentile(window_data, 0.75)
        IQR = Q3 - Q1

        # 定义局部离群值
        if arr[i] < Q1 - threshold * IQR or arr[i] > Q3 + threshold * IQR:
            # 用邻近非离群值替换
            non_outlier_data = window_data[(window_data >= Q1 - threshold * IQR) & (window_data <= Q3 + threshold * IQR)]
            if len(non_outlier_data) > 0:
                arr[i] = np.mean(non_outlier_data)

    return arr

def smooth_data(arr, window_parameter=31, polyorder_parameter=2): # 平滑数据
    arr_smoothed = savgol_filter(arr, window_length=window_parameter, polyorder=polyorder_parameter)
    return arr_smoothed

def extract_features(sequence, slice_num=10):  # 把序列切成slice_num段，每段取均值、最大值、最小值、方差，共40个特征，返回一个拼接的一维数组
    # 计算每个子序列的基本长度
    n = len(sequence)
    sub_seq_length = n // slice_num if n % slice_num == 0 else n // slice_num + 1# 向上取整
    remainder = sub_seq_length - (n // slice_num + 1) * slice_num + n # 处理最后一段未填充满

    # 初始化特征数组
    features_mean = []
    features_max = []
    features_min = []
    features_var = []
    start = 0

    # 对每个子序列进行迭代
    for i in range(slice_num):
        # 调整子序列长度
        end = start
        if i < slice_num - 1:
            end = start + sub_seq_length
        else:
            end = start + remainder if remainder > 0 else sub_seq_length + start
        # print("start" + str(start) + " end" + str(end))
        sub_seq = sequence[start:end]

        # 计算特征
        mean = np.mean(sub_seq)
        max_value = np.max(sub_seq)
        min_value = np.min(sub_seq)
        variance = np.var(sub_seq)

        # 添加到特征数组
        features_mean.append(mean)
        features_max.append(max_value)
        features_min.append(min_value)
        features_var.append(variance)

        # 更新起始位置
        start = end

    return np.concatenate([features_mean, features_max, features_min, features_var])

def difference_gaze_lr_euler_angle(user, date, num): # 读取用户特定日期和序号的视线数据，以3个list分别返回左右视线Yaw, Pitch, Roll角度的差异, num从1开始
    data = pd.read_csv(os.path.join(os.getcwd(), "data/unity_processed_data", "GazeCalculate_data_" + user + "-" + date + "-" + str(num) + "_unity_processed.csv"))
    # Create DataFrame
    df = pd.DataFrame(data)

    L_R_Yaw = [x - y if abs(x - y) < 180 else (x - y - 360 if x - y >180 else x - y +360) for x, y in zip(df['L_Yaw'], df['R_Yaw'])]
    L_R_Pitch = [x - y if abs(x - y) < 180 else (x - y - 360 if x - y >180 else x - y +360) for x, y in zip(df['L_Pitch'], df['R_Pitch'])]
    L_R_Roll = [x - y if abs(x - y) < 180 else (x - y - 360 if x - y >180 else x - y +360) for x, y in zip(df['L_Roll'], df['R_Roll'])]
    return L_R_Yaw, L_R_Pitch, L_R_Roll

def difference_gaze_head(user, date, num, eye='L', angle='Yaw'):# 读取用户特定日期和序号的视线数据和头部数据，以list返回视线和头部偏航角度之间的差异, num从1开始, eye='L' or 'R', angle='Yaw' or 'Pitch' or 'Roll'
    if eye not in ['L', 'R']:
        raise ValueError("eye must be 'L' or 'R'")
    if angle not in ['Yaw', 'Pitch', 'Roll']:
        raise ValueError("angle must be 'Yaw' or 'Pitch' or 'Roll'")
    # 数据存储在unity_processed_data目录下
    data1 = pd.read_csv(os.path.join("data/unity_processed_data", "GazeCalculate_data_" + user + "-" + date + "-" + str(num) + "_unity_processed.csv"))
    data2 = pd.read_csv(os.path.join("data/unity_processed_data", "Head_data_" + user + "-" + date + "-" + str(num) + "_unity_processed.csv"))

    # 将读取的数据转换为DataFrame
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)

    # 从视线数据中计算初始偏航角度的偏移
    gaze_zero = df1[eye + '_' + angle][0]
    gaze_df = [x - gaze_zero if abs(x - gaze_zero) < 200 else (x - gaze_zero - 360 if x - gaze_zero > 200 else x - gaze_zero + 360) for x in df1[eye + '_' + angle]]

    # 从头部数据中计算初始偏航角度（Yaw）的偏移，并调整角度超过180度的情况
    head_zero = df2[angle][0] if df2[angle][0] < 180 else df2[angle][0] - 360
    head_df = [x - head_zero if x < 180 else x - head_zero - 360 for x in df2[angle]]

    # 返回视线和头部偏航角度之间的差异
    return [x - y for x, y in zip(gaze_df, head_df)]

def fourier_gaze(user, date, num, eye='L', angle='Yaw'): # 读取用户特定日期和序号的视线数据，以list返回视线偏航角度的傅里叶变换结果
    if eye not in ['L', 'R']:
        raise ValueError("eye must be 'L' or 'R'")
    if angle not in ['Yaw', 'Pitch', 'Roll']:
        raise ValueError("angle must be 'Yaw' or 'Pitch' or 'Roll'")
    data1 = pd.read_csv(os.path.join("data/unity_processed_data","GazeCalculate_data_" + user + "-" + date + "-" + str(num+1) + "_unity_processed.csv"))
    df1 = pd.DataFrame(data1)


    gaze_zero = df1[eye + '_' + angle][0]
    gaze_df = [x - gaze_zero if abs(x - gaze_zero) < 200 else (x - gaze_zero -  360 if x- gaze_zero >200 else x - gaze_zero + 360 ) for x in df1[eye + '_' + angle]]
    # gaze_df_slice = gaze_df[slice_l:slice_r]
    fft_gaze = np.fft.fft(gaze_df)
    freq_gaze=np.fft.fftfreq(len(gaze_df),0.02)

    return gaze_df, fft_gaze, freq_gaze  # 返回视线偏航角度的傅里叶变换结果

def quaternion_to_euler (x, y, z, w): # result is different from Unity, idk why
    # x, y, z, w are numpy arrays of shape (n,)
    # return three numpy arrays of shape (n,) representing roll, pitch, yaw
    
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2 (t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip (t2, -1.0, 1.0) # avoid invalid values due to numerical errors
    pitch = np.arcsin (t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2 (t3, t4)

    return np.degrees([roll, pitch, yaw])

def quaternion_to_euler_df(dataframe): # result is different from Unity, idk why
    """
    Convert quaternions in a DataFrame to euler angles.
    Assumes columns named 'L-QuaternionW', 'L-QuaternionX', 'L-QuaternionY', 'L-QuaternionZ'.
    Returns a new DataFrame with columns 'Yaw', 'Pitch', 'Roll'.
    """
    def single_quaternion_to_euler(w, x, y, z):
        # Roll (X-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (Y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)

        # Yaw (Z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.degrees([yaw, pitch, roll])  # Convert to degrees

    euler_angles = dataframe.apply(lambda row: single_quaternion_to_euler(
        row['L-QuaternionW'], row['L-QuaternionX'], row['L-QuaternionY'], row['L-QuaternionZ']), axis=1)

    return pd.DataFrame(euler_angles.tolist(), columns=['Yaw', 'Pitch', 'Roll'])

def unity_quaternion_to_euler(x, y, z, w): # result is different from Unity, idk why
    """
    Convert a Unity-style quaternion into euler angles (in degrees)
    """
    # Roll (X-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (Y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)  # use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)

    # Yaw (Z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.degrees([yaw, pitch, roll])  # Convert to degrees

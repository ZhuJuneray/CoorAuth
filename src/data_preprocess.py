import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from cycler import cycler
import os
import json
import itertools

def read_data_name_from_json(filepath = os.path.join(os.getcwd(), "src/data.json")):
        with open(filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)
        data_list = [f"{item['studytype']}-{item['names']}-{item['date']}" for item in data['data']]
        return data_list

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
        Q1 = np.percentile(window_data, 25)
        Q3 = np.percentile(window_data, 75)
        IQR = Q3 - Q1

        # 定义局部离群值
        if arr[i] < Q1 - threshold * IQR or arr[i] > Q3 + threshold * IQR:
            # 用邻近非离群值替换
            non_outlier_data = window_data[(window_data >= Q1 - threshold * IQR) & (window_data <= Q3 + threshold * IQR)]
            if len(non_outlier_data) > 0:
                arr[i] = np.mean(non_outlier_data)

    return arr


def smooth_data(arr, window_parameter=9, polyorder_parameter=2): # 平滑数据
    arr_smoothed = savgol_filter(arr, window_length=window_parameter, polyorder=polyorder_parameter)
    return arr_smoothed

def extract_features(sequence, slice_num=10):  # 把序列切成十段，每段取均值、最大值、最小值、方差，共40个特征，返回一个拼接的一维数组
    # 计算每个子序列的基本长度和额外长度
    n = len(sequence)
    # print("length" + str(n))
    sub_seq_length = n // slice_num if n % slice_num == 0 else n // slice_num + 1# 向上取整
    remainder = sub_seq_length - (n // slice_num + 1) * slice_num + n # 处理最后一段未填充满

    # 初始化特征数组
    features = []
    features_mean = []       # 均值
    features_max = []        # 最大值
    features_min = []        # 最小值
    features_var = []        # 方差
    features_median = []     # 中位数
    features_rms = []        # 均方根值
    features_std = []        # 标准差
    features_mad = []        # 平均绝对偏差
    features_kurtosis = []   # 峰度
    features_skewness = []   # 偏度
    features_iqr = []        # 四分位数范围
    # features_roughness = []  # 粗糙度，需要具体定义
    # features_sharpness = []  # 锋利度，需要具体定义
    features_mc = []         # 均值穿越次数
    features_wamp = []       # Willison幅度
    features_ssc = []        # 坡度符号变化次数
    start = 0

    # 对每个子序列进行迭代
    for i in range(slice_num):
        # 调整子序列长度
        end = start + sub_seq_length if i < slice_num - 1 else start + (remainder if remainder > 0 else sub_seq_length)
        sub_seq = sequence[start:end]

        # 计算特征
        mean = np.mean(sub_seq)                         # 计算均值
        max_value = np.max(sub_seq)                     # 计算最大值
        min_value = np.min(sub_seq)                     # 计算最小值
        variance = np.var(sub_seq)                      # 计算方差
        median = np.median(sub_seq)                     # 计算中位数
        rms = np.sqrt(np.mean(np.square(sub_seq)))      # 计算均方根
        std_dev = np.std(sub_seq)                       # 计算标准差
        mad = np.mean(np.abs(sub_seq - np.mean(sub_seq))) # 计算平均绝对偏差
        # kurt = kurtosis(sub_seq)                        # 计算峰度
        # skewness = skew(sub_seq)                        # 计算偏度
        q75, q25 = np.percentile(sub_seq, [75, 25])
        iqr = q75 - q25                                 # 计算四分位数范围
        mc = np.sum(np.sign(sub_seq[:-1]) != np.sign(sub_seq[1:])) / len(sub_seq) # 计算均值穿越次数
        threshold = 0.1  # 根据需要设置阈值
        wamp = np.sum(np.abs(np.diff(sub_seq)) > threshold) # 计算Willison幅度
        ssc = np.sum(np.diff(np.sign(np.diff(sub_seq))) != 0) # 计算坡度符号变化次数

        # 添加到特征数组
        features_mean.append(mean)
        features_max.append(max_value)
        features_min.append(min_value)
        features_var.append(variance)
        features_median.append(median)
        features_rms.append(rms)
        features_std.append(std_dev)
        features_mad.append(mad)
        # features_kurtosis.append(kurt)
        # features_skewness.append(skewness)
        features_iqr.append(iqr)
        # features_roughness.append(roughness)  # 根据定义实现
        # features_sharpness.append(sharpness)  # 根据定义实现
        features_mc.append(mc)
        features_wamp.append(wamp)
        features_ssc.append(ssc)

        # 更新起始位置
        start = end

    return np.concatenate([features_mean, features_max, features_min, features_var,
                           features_median, features_rms, features_std, features_mad,
                            features_iqr,
                           features_mc, features_wamp, features_ssc])

def difference_gaze_lr_euler_angle(user, date, num): # 读取用户特定日期和序号的视线数据，以3个list分别返回左右视线Yaw, Pitch, Roll角度的差异, num从1开始
    data = pd.read_csv(os.path.join(os.getcwd(), "data", "GazeCalculate_data_" + user + "-" + date + "-" + str(num) + "_unity_processed.csv"))
    # Create DataFrame
    df = pd.DataFrame(data)

    L_R_Yaw = [x - y if abs(x - y) < 180 else (x - y - 360 if x - y >180 else x - y +360) for x, y in zip(df['L_Yaw'], df['R_Yaw'])]
    L_R_Pitch = [x - y if abs(x - y) < 180 else (x - y - 360 if x - y >180 else x - y +360) for x, y in zip(df['L_Pitch'], df['R_Pitch'])]
    L_R_Roll = [x - y if abs(x - y) < 180 else (x - y - 360 if x - y >180 else x - y +360) for x, y in zip(df['L_Roll'], df['R_Roll'])]
    return L_R_Yaw, L_R_Pitch, L_R_Roll

def difference_gaze_head(member, size, pin, num, eye='L', angle='Yaw', rotdir = "", noise_flag=False, noise_level=0.1):# 读取用户特定日期和序号的视线数据和头部数据，以list返回视线和头部偏航角度之间的差异, num从1开始, eye='L' or 'R', angle='Yaw' or 'Pitch' or 'Roll'
    if eye not in ['L', 'R']:
        raise ValueError("eye must be 'L' or 'R'")
    if angle not in ['Yaw', 'Pitch', 'Roll']:
        raise ValueError("angle must be 'Yaw' or 'Pitch' or 'Roll'")
    # 数据存储在unity_processed_data目录下
    
    data1 = pd.read_csv(os.path.join(rotdir, f"VRAuthStudy1Angle-{member.split('-')[2]}/P{member.split('-')[1]}/GazeCalculate_data_{member.split('-')[0]}-{member.split('-')[1]}-{member.split('-')[2]}-{str(size)}-{str(pin)}-{str(num)}.csv"))
    data2 = pd.read_csv(os.path.join(rotdir, f"VRAuthStudy1Angle-{member.split('-')[2]}/P{member.split('-')[1]}/Head_data_{member.split('-')[0]}-{member.split('-')[1]}-{member.split('-')[2]}-{str(size)}-{str(pin)}-{str(num)}.csv"))

    # 将读取的数据转换为DataFrame
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)

    # 从视线数据中计算初始偏航角度的偏移
    if noise_flag:
        df1[eye + '_' + angle]=add_noise(df1[eye + '_' + angle], noise_level=noise_level)
    gaze_zero = np.mean(df1[eye + '_' + angle][0:5])
    gaze_df = [x - gaze_zero if abs(x - gaze_zero) < 200 else (x - gaze_zero - 360 if x - gaze_zero > 200 else x - gaze_zero + 360) for x in df1[eye + '_' + angle]]

    # 从头部数据中计算初始偏航角度（Yaw）的偏移，并调整角度超过180度的情况
    if noise_flag:
        df2[angle]=add_noise(df2[angle], noise_level=noise_level)
    head_zero = np.mean([x if x < 180 else x - 360 for x in df2[angle][0:5]])
    head_df = [x - head_zero if x < 180 else x - head_zero - 360 for x in df2[angle]]

    # 返回视线和头部偏航角度之间的差异
    return [x - y for x, y in zip(gaze_df, head_df)]

def fourier_gaze(user, date, num, eye='L', angle='Yaw'): # 读取用户特定日期和序号的视线数据，以list返回视线偏航角度的傅里叶变换结果
    if eye not in ['L', 'R']:
        raise ValueError("eye must be 'L' or 'R'")
    if angle not in ['Yaw', 'Pitch', 'Roll']:
        raise ValueError("angle must be 'Yaw' or 'Pitch' or 'Roll'")
    data1 = pd.read_csv(os.path.join("data","GazeCalculate_data_" + user + "-" + date + "-" + str(num+1) + "_unity_processed.csv"))
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

# def read_data_name_from_json(filepath):
#     with open(filepath, 'r', encoding='utf-8') as file:
#         data = json.load(file)

#     train_set=[]
#     train_set_positive_label = []
#     train_set_negative_label = []
#     test_set = []

#     # 处理 train_set
#     # if data["scene"] and data["train_set_scene"]:
#     #         raise Exception("scene and train_set_scene can't be both set")
#     for item in data['train_set']['positive_label']:
#         names = [item['names']] if isinstance(item['names'], str) else item['names']
#         dates = [item['date']] if isinstance(item['date'], str) else item['date']
#         range_ = item['range'] if item['range'] else 'all'
#         namess=[]
#         if data["scene"]:
#             for name in names:
#                 namess.append(data["scene"]+name)
#         elif data["train_set_scene"]:
#             for name in names:
#                 namess.append(data["train_set_scene"]+name)
#         else:
#             namess=names
#         for name, date in itertools.product(namess, dates):
#             train_set.append([name, date, range_])
#             train_set_positive_label.append([name, date, range_])
    
#     for item in data['train_set']['negative_label']:
#         names = [item['names']] if isinstance(item['names'], str) else item['names']
#         dates = [item['date']] if isinstance(item['date'], str) else item['date']
#         range_ = item['range'] if item['range'] else 'all'
#         namess=[]
#         if data["scene"]:
#             for name in names:
#                 namess.append(data["scene"]+name)
#         elif data["train_set_scene"]:
#             for name in names:
#                 namess.append(data["train_set_scene"]+name)
#         else:
#             namess=names
#         for name, date in itertools.product(namess, dates):
#             train_set.append([name, date, range_])
#             train_set_negative_label.append([name, date, range_])

#     # 处理 test_set
#     if data["scene"] and data["test_set_scene"]:
#             raise Exception("scene and test_set_scene can't be both set")
#     for item in data['test_set']:
#         names = [item['names']] if isinstance(item['names'], str) else item['names']
#         dates = [item['date']] if isinstance(item['date'], str) else item['date']
#         range_ = item['range'] if item['range'] else 'all'
#         namess=[]
#         if data["scene"]:
#             for name in names:
#                 namess.append(data["scene"]+name)
#         elif data["test_set_scene"]:
#             for name in names:
#                 namess.append(data["test_set_scene"]+name)
#         else:
#             namess=names
#         for name, date in itertools.product(namess, dates):
#             test_set.append([name, date, range_])

#     return train_set, train_set_positive_label, train_set_negative_label, test_set

def range_to_int_value(range_str):
        def range_to_int_start_end(range_str, value='start'):
            values = list(map(int, range_str.split('-')))
            return values[0] if value == 'start' else values[1]
        return range_to_int_start_end(range_str, 'end')-range_to_int_start_end(range_str, 'start')

def google_sheet_to_json(studytype = "study1", credential_path = "src/credentials.json", google_sheet_name = "被试招募", json_save_path= "src/data.json"):
    import gspread
    def map_names_to_numbers(names):
        name_to_number = {}
        number_list = []
        counter = 1
        for name in names:
            if name not in name_to_number:
                name_to_number[name] = counter
                counter += 1
            number_list.append(name_to_number[name])
        return number_list

    client = gspread.service_account(filename=credential_path)
    spreadsheet = client.open(google_sheet_name)
    sheet = spreadsheet.sheet1
    # Fetch the first column values
    first_column = sheet.col_values(1)
    # for participant in range(count_study):
    first_occurrence = None
    last_occurrence = None

    for i, value in enumerate(first_column, start=1):  # start=1 to start counting from row 1
        if value == studytype:
            last_occurrence = i
            if first_occurrence is None:
                first_occurrence = i

    data_range = f"D{first_occurrence}:D{last_occurrence}"
    column_data = sheet.range(data_range)
    names = [cell.value for cell in column_data if cell.value.strip()]
    numbered_list = map_names_to_numbers(names)

    data_list = []

    for i in range(last_occurrence-first_occurrence+1):  # Adjust the range as needed
        # Generate or collect your data
        data_item = {"studytype" : studytype, "names": numbered_list[i], "date": sheet.col_values(2)[i+first_occurrence-1]}
        # Append the data item to the list
        data_list.append(data_item)

    # Wrap the list in a dictionary under the key 'data'
    data_to_write = {"data": data_list, "latter_auth": []}

    # Write the data to a JSON file
    with open(json_save_path, 'w', encoding='utf-8') as file:
        json.dump(data_to_write, file, ensure_ascii=False, indent=4)

def add_noise(data, noise_level=0.1):
    std_dev = noise_level * np.std(data)  # 计算噪声的标准差
    noise = np.random.normal(0, std_dev, data.shape)  # 生成高斯噪声
    data_noisy = data + noise  # 将噪声添加到原始数据
    return data_noisy

def data_zero_smooth_feature(eye_data_dir=None, head_data_dir =None, noise_flag=False, noise_level=0.1):
    data_head = pd.read_csv(head_data_dir)
    QuaternionX_data = data_head['H-QuaternionX']
    if noise_flag:
        QuaternionX_data = add_noise(QuaternionX_data, noise_level)
    QuaternionX_data = QuaternionX_data - np.mean(QuaternionX_data[0:5])
    QuaternionX_data_smoothed = smooth_data(QuaternionX_data)
    d1 = np.array(QuaternionX_data_smoothed)
    d1_feat = extract_features(d1)
    QuaternionY_data = data_head['H-QuaternionY']
    if noise_flag:
        QuaternionY_data = add_noise(QuaternionY_data, noise_level)
    QuaternionY_data = QuaternionY_data - np.mean(QuaternionY_data[0:5])
    QuaternionY_data_smoothed = smooth_data(QuaternionY_data)
    d2 = np.array(QuaternionY_data_smoothed)
    d2_feat = extract_features(d2)
    QuaternionZ_data = data_head['H-QuaternionZ']
    if noise_flag:
        QuaternionZ_data = add_noise(QuaternionZ_data, noise_level)
    QuaternionZ_data = QuaternionZ_data - np.mean(QuaternionZ_data[0:5])
    QuaternionZ_data_smoothed = smooth_data(QuaternionZ_data)
    d3 = np.array(QuaternionZ_data_smoothed)
    d3_feat = extract_features(d3)
    QuaternionW_data = data_head['H-QuaternionW']
    if noise_flag:
        QuaternionW_data = add_noise(QuaternionW_data, noise_level)
    QuaternionW_data = QuaternionW_data - np.mean(QuaternionW_data[0:5])
    QuaternionW_data_smoothed = smooth_data(QuaternionW_data)
    d4 = np.array(QuaternionW_data_smoothed)
    d4_feat = extract_features(d4)

    Vector3X_data = data_head['H-Vector3X']
    if noise_flag:
        Vector3X_data = add_noise(Vector3X_data, noise_level)
    Vector3X_data = Vector3X_data - np.mean(Vector3X_data[0:5])
    Vector3X_data_smoothed = smooth_data(Vector3X_data)
    v1 = np.array(Vector3X_data_smoothed)
    v1_feat = extract_features(v1)
    Vector3Y_data = data_head['H-Vector3Y']
    if noise_flag:
        Vector3Y_data = add_noise(Vector3Y_data, noise_level)
    Vector3Y_data = Vector3Y_data - np.mean(Vector3Y_data[0:5])
    Vector3Y_data_smoothed = smooth_data(Vector3Y_data)
    v2 = np.array(Vector3Y_data_smoothed)
    v2_feat = extract_features(v2)
    Vector3Z_data = data_head['H-Vector3Z']
    if noise_flag:
        Vector3Z_data = add_noise(Vector3Z_data, noise_level)
    Vector3Z_data = Vector3Z_data - np.mean(Vector3Z_data[0:5])
    Vector3Z_data_smoothed = smooth_data(Vector3Z_data)
    v3 = np.array(Vector3Z_data_smoothed)
    v3_feat = extract_features(v3)

    # Eye points
    data_eye = pd.read_csv(eye_data_dir)
    QuaternionX_data = data_eye['L-QuaternionX']
    if noise_flag:
        QuaternionX_data = add_noise(QuaternionX_data, noise_level)
    QuaternionX_data = QuaternionX_data - np.mean(QuaternionX_data[0:5])
    QuaternionX_data_smoothed = smooth_data(QuaternionX_data)
    d1_el = np.array(QuaternionX_data_smoothed)
    d1_el_feat = extract_features(d1_el)
    QuaternionY_data = data_eye['L-QuaternionY']
    if noise_flag:
        QuaternionY_data = add_noise(QuaternionY_data, noise_level)
    QuaternionY_data = QuaternionY_data - np.mean(QuaternionY_data[0:5])
    QuaternionY_data_smoothed = smooth_data(QuaternionY_data)
    d2_el = np.array(QuaternionY_data_smoothed)
    d2_el_feat = extract_features(d2_el)
    QuaternionZ_data = data_eye['L-QuaternionZ']
    if noise_flag:
        QuaternionZ_data = add_noise(QuaternionZ_data, noise_level)
    QuaternionZ_data = QuaternionZ_data - np.mean(QuaternionZ_data[0:5])
    QuaternionZ_data_smoothed = smooth_data(QuaternionZ_data)
    d3_el = np.array(QuaternionZ_data_smoothed)
    d3_el_feat = extract_features(d3_el)
    QuaternionW_data = data_eye['L-QuaternionW']
    if noise_flag:
        QuaternionW_data = add_noise(QuaternionW_data, noise_level)
    QuaternionW_data = QuaternionW_data - np.mean(QuaternionW_data[0:5])
    QuaternionW_data_smoothed = smooth_data(QuaternionW_data)
    d4_el = np.array(QuaternionW_data_smoothed)
    d4_el_feat = extract_features(d4_el)

    QuaternionX_data = data_eye['R-QuaternionX']
    if noise_flag:
        QuaternionX_data = add_noise(QuaternionX_data, noise_level)
    QuaternionX_data = QuaternionX_data - np.mean(QuaternionX_data[0:5])
    QuaternionX_data_smoothed = smooth_data(QuaternionX_data)
    d1_er = np.array(QuaternionX_data_smoothed)
    d1_er_feat = extract_features(d1_er)
    QuaternionY_data = data_eye['R-QuaternionY']
    if noise_flag:
        QuaternionY_data = add_noise(QuaternionY_data, noise_level)
    QuaternionY_data = QuaternionY_data - np.mean(QuaternionY_data[0:5])
    QuaternionY_data_smoothed = smooth_data(QuaternionY_data)
    d2_er = np.array(QuaternionY_data_smoothed)
    d2_er_feat = extract_features(d2_er)
    QuaternionZ_data = data_eye['R-QuaternionZ']
    if noise_flag:
        QuaternionZ_data = add_noise(QuaternionZ_data, noise_level)
    QuaternionZ_data = QuaternionZ_data - np.mean(QuaternionZ_data[0:5])
    QuaternionZ_data_smoothed = smooth_data(QuaternionZ_data)
    d3_er = np.array(QuaternionZ_data_smoothed)
    d3_er_feat = extract_features(d3_er)
    QuaternionW_data = data_eye['R-QuaternionW']
    if noise_flag:
        QuaternionW_data = add_noise(QuaternionW_data, noise_level)
    QuaternionW_data = QuaternionW_data - np.mean(QuaternionW_data[0:5])
    QuaternionW_data_smoothed = smooth_data(QuaternionW_data)
    d4_er = np.array(QuaternionW_data_smoothed)
    d4_er_feat = extract_features(d4_er)

    return d1, d1_feat, d2, d2_feat, d3, d3_feat, d4, d4_feat, v1, v1_feat, v2, v2_feat, v3, v3_feat, d1_el, d1_el_feat, d2_el, d2_el_feat, d3_el, d3_el_feat, d4_el, d4_el_feat, d1_er, d1_er_feat, d2_er, d2_er_feat, d3_er, d3_er_feat, d4_er, d4_er_feat

def merged_array_generator(member, size, pin, num, model, rotdir, noise_flag = None, noise_level=0.1):
    d1, d1_feat, d2, d2_feat, d3, d3_feat, d4, d4_feat, v1, v1_feat, v2, v2_feat, v3, v3_feat, d1_el, d1_el_feat, d2_el, d2_el_feat, d3_el, d3_el_feat, d4_el, d4_el_feat, d1_er, d1_er_feat, d2_er, d2_er_feat, d3_er, d3_er_feat, d4_er, d4_er_feat = data_zero_smooth_feature(head_data_dir=rotdir + f"VRAuthStudy1-{member.split('-')[2]}/P{member.split('-')[1]}/Head_data_{member}-{str(size)}-{str(pin)}-{str(num+1)}.csv", eye_data_dir=rotdir + f"VRAuthStudy1-{member.split('-')[2]}/P{member.split('-')[1]}/GazeRaw_data_{member}-{str(size)}-{str(pin)}-{str(num+1)}.csv", noise_flag=noise_flag, noise_level=noise_level)
    # Head and eye points
    diff_yaw_data = difference_gaze_head(member, size, pin, num+1, rotdir=rotdir, noise_flag=noise_flag, noise_level=noise_level)
    diff_yaw_smooth = smooth_data(diff_yaw_data, window_parameter=9)
    dy_el_feat = extract_features(np.array(diff_yaw_smooth))
    diff_pitch_data = difference_gaze_head(member, size, pin, num+1, eye='L', angle='Pitch', rotdir=rotdir, noise_flag=noise_flag, noise_level=noise_level)
    diff_pitch_smooth = smooth_data(diff_pitch_data, window_parameter=9)
    dp_el_feat = extract_features(np.array(diff_pitch_smooth))
    diff_roll_data = difference_gaze_head(member, size, pin, num+1, eye='L', angle='Roll', rotdir=rotdir, noise_flag=noise_flag, noise_level=noise_level)
    diff_roll_smooth = smooth_data(diff_roll_data, window_parameter=9)
    dr_el_feat = extract_features(np.array(diff_roll_smooth))

    if model == 'head':
        # merged_array = np.concatenate(
        #     [d1_feat, d2_feat, d3_feat, d4_feat, v1_feat, v2_feat, v3_feat])
        merged_array = np.concatenate(
            [d1_feat, d2_feat, d3_feat, d4_feat])
    # 利用特征：切10段的特征
    elif model == "eye":
        merged_array = np.concatenate(
            [d1_el_feat, d2_el_feat, d3_el_feat,
                d4_el_feat, d1_er_feat, d2_er_feat, d3_er_feat, d4_er_feat, ])
    elif model == "head+eye":
        # 利用特征：切10段的特征
        merged_array = np.concatenate(
            [d1_feat, d2_feat, d3_feat, d4_feat, v1_feat, v2_feat, v3_feat, d1_el_feat, d2_el_feat,
                d3_el_feat,
                d4_el_feat, d1_er_feat, d2_er_feat, d3_er_feat, d4_er_feat, ])
    elif model == "diff":
        # 利用特征：切10段的特征
        merged_array = np.concatenate([dy_el_feat, dp_el_feat, dr_el_feat]
        )
    elif model == "eye+diff":
        merged_array = np.concatenate([dy_el_feat, dp_el_feat,
                                        dr_el_feat, d1_el_feat, d2_el_feat, d3_el_feat,
                d4_el_feat, d1_er_feat, d2_er_feat, d3_er_feat, d4_er_feat]
        )
    else:
        merged_array = np.concatenate(
            [d1_feat, d2_feat, d3_feat, d4_feat, v1_feat, v2_feat, v3_feat, d1_el_feat, d2_el_feat,
                d3_el_feat,
                d4_el_feat, d1_er_feat, d2_er_feat, d3_er_feat, d4_er_feat,
                dy_el_feat, dp_el_feat, dr_el_feat])
    # print(d1)
    if np.isnan(d1_feat).any():
        print("NaN values found in d1_feat")
        # 定位NaN值
        print(np.argwhere(np.isnan(d1_feat)))
    # if num == 1:
    #     print("user" + user + "data" + str(merged_array))
    return merged_array

# read_data_name_from_json()
# google_sheet_to_json(studytype="study2")
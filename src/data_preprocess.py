import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import kurtosis, skew
from cycler import cycler
import os
import json
import itertools
import warnings


def read_data_latter_data_json(filepath="D:\pycharm\srt_vr_auth\src\data.json"):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    data_list = [f"{item['studytype']}_{item['names']}_{item['date']}_{item['num_range']}" for item in data['data']]
    latter_auth_list = [f"{item['studytype']}_{item['names']}_{item['date']}_{item['num_range']}" for item in
                        data['latter_auth']]
    return data_list, latter_auth_list


def read_data_name_from_json(filepath=os.path.join(os.getcwd(), "src/data.json")):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    data_list = [f"{item['studytype']}-{item['names']}-{item['date']}" for item in data['data']]
    return data_list


def replace_local_outliers(arr, window_size=5, threshold=1.5):  # 去除离群值

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
            non_outlier_data = window_data[
                (window_data >= Q1 - threshold * IQR) & (window_data <= Q3 + threshold * IQR)]
            if len(non_outlier_data) > 0:
                arr[i] = np.mean(non_outlier_data)

    return arr


# 1231 update 考虑smooth_data的细节：是否需要平滑？是否需要去除离群值？
def smooth_data(arr, window_parameter=9, polyorder_parameter=2):  # 平滑数据
    arr = replace_local_outliers(arr)
    # arr = savgol_filter(arr, window_length=window_parameter, polyorder=polyorder_parameter)
    return arr


# 1231update ranges为fixation的一个list
def extract_features(sequence, slice_num=10, ranges=None):  # 把序列切成十段，每段取均值、最大值、最小值、方差，共40个特征，返回一个拼接的一维数组
    # 如果range不空，则按照range中的start和end切分，saccades占slice_num - n段，fixation占n段
    range_fixation = []
    range_sacaades = []
    fixation_num = 5
    saccades_num = slice_num - fixation_num  # 每种切段的数量
    tmp_end = 0
    if ranges is not None:
        for start, end in ranges:
            range_fixation.append([start, end])
            range_sacaades.append([tmp_end, start]) if tmp_end != 0 else None
            tmp_end = end
        # 长度不足, 使用已经添加过的数据来填充 fixation使用第一段填充，saccades用最后一段填充
        while len(range_fixation) < fixation_num:
            # 从 range_fixation 中获取数据填充
            previous_data = range_fixation[0]
            range_fixation.append(previous_data)
        while len(range_sacaades) < saccades_num:
            previous_data = range_sacaades[-1]
            range_sacaades.append(previous_data)
    # range 为空，等距切分
    else:
        # 计算每个子序列的基本长度和额外长度
        n = len(sequence)
        # print("length" + str(n))
        sub_seq_length = n // slice_num if n % slice_num == 0 else n // slice_num + 1  # 向上取整
        remainder = sub_seq_length - (n // slice_num + 1) * slice_num + n  # 处理最后一段未填充满
        start = 0
        for i in range(slice_num):
            # 调整子序列长度
            end = start + sub_seq_length if i < slice_num - 1 else start + (
                remainder if remainder > 0 else sub_seq_length)
            range_sacaades.append([start, end])
            range_fixation.append([start, end])
            start = end

    # 处理超长的情况，saccades保留最后，fixation保留前面
    if len(range_sacaades) > saccades_num:
        range_sacaades = range_sacaades[-saccades_num:]
    if len(range_fixation) > fixation_num:
        range_fixation = range_fixation[:fixation_num]
    # print(ranges, "range_sacaades", range_sacaades, "range_fix", range_fixation)
    ranges = range_fixation + range_sacaades  # 也即5个fixation和5个saccades
    # print("changed ranges", ranges)

    # update 1.1 改成了函数，获得序列本身的特征向量
    seq_initial = get_n_derivation_features(sequence, ranges)
    # 1阶导
    seq_second = get_n_derivation_features(np.diff(sequence), ranges)

    seq_all = np.concatenate([seq_initial])
    return seq_all


# update1.1 获得n阶导的特征向量
def get_n_derivation_features(sequence, ranges):
    # 初始化特征数组
    features = []
    features_mean = []  # 均值
    features_max = []  # 最大值
    features_min = []  # 最小值
    features_var = []  # 方差
    features_median = []  # 中位数
    features_rms = []  # 均方根值
    features_std = []  # 标准差
    features_mad = []  # 平均绝对偏差
    features_kurtosis = []  # 峰度
    features_skewness = []  # 偏度
    features_iqr = []  # 四分位数范围
    # features_roughness = []  # 粗糙度，需要具体定义
    # features_sharpness = []  # 锋利度，需要具体定义
    features_mc = []  # 均值穿越次数
    features_wamp = []  # Willison幅度
    features_ssc = []  # 坡度符号变化次数

    for start, end in ranges:
        sub_seq = sequence[start:end]
        # 计算特征
        mean = np.mean(sub_seq)  # 计算均值
        max_value = np.max(sub_seq)  # 计算最大值
        min_value = np.min(sub_seq)  # 计算最小值
        variance = np.var(sub_seq)  # 计算方差
        median = np.median(sub_seq)  # 计算中位数
        rms = np.sqrt(np.mean(np.square(sub_seq)))  # 计算均方根
        std_dev = np.std(sub_seq)  # 计算标准差
        mad = np.mean(np.abs(sub_seq - np.mean(sub_seq)))  # 计算平均绝对偏差
        kurt = kurtosis(sub_seq)  # 计算峰度
        skewness = skew(sub_seq)  # 计算偏度
        q75, q25 = np.percentile(sub_seq, [75, 25])
        iqr = q75 - q25  # 计算四分位数范围
        mc = np.sum(np.sign(sub_seq[:-1]) != np.sign(sub_seq[1:])) / len(sub_seq)  # 计算均值穿越次数
        threshold = 0.1  # 根据需要设置阈值
        wamp = np.sum(np.abs(np.diff(sub_seq)) > threshold)  # 计算Willison幅度
        ssc = np.sum(np.diff(np.sign(np.diff(sub_seq))) != 0)  # 计算坡度符号变化次数

        # 添加到特征数组
        features_mean.append(mean)  # high
        features_max.append(max_value)  # high
        features_min.append(min_value)  # high
        features_var.append(variance)   
        features_median.append(median)  # high
        features_rms.append(rms)        
        features_std.append(std_dev)    
        features_mad.append(mad)        
        features_kurtosis.append(kurt)  # low
        features_skewness.append(skewness)  # low
        features_iqr.append(iqr)  # mid
        # features_roughness.append(roughness)  #
        # features_sharpness.append(sharpness)  #
        features_mc.append(mc)  # low
        features_wamp.append(wamp)  # zero
        features_ssc.append(ssc)  # low

    return np.concatenate([features_mean, features_max, features_min, features_var, features_median, features_rms, 
                            features_std, features_mad, features_kurtosis, features_skewness, features_iqr,
                            features_mc, features_wamp, features_ssc])


def difference_gaze_lr_euler_angle(user, date, num):  # 读取用户特定日期和序号的视线数据，以3个list分别返回左右视线Yaw, Pitch, Roll角度的差异, num从1开始
    data = pd.read_csv(os.path.join(os.getcwd(), "data", "GazeCalculate_data_" + user + "-" + date + "-" + str(
        num) + "_unity_processed.csv"))
    # Create DataFrame
    df = pd.DataFrame(data)

    L_R_Yaw = [x - y if abs(x - y) < 180 else (x - y - 360 if x - y > 180 else x - y + 360) for x, y in
               zip(df['L_Yaw'], df['R_Yaw'])]
    L_R_Pitch = [x - y if abs(x - y) < 180 else (x - y - 360 if x - y > 180 else x - y + 360) for x, y in
                 zip(df['L_Pitch'], df['R_Pitch'])]
    L_R_Roll = [x - y if abs(x - y) < 180 else (x - y - 360 if x - y > 180 else x - y + 360) for x, y in
                zip(df['L_Roll'], df['R_Roll'])]
    return L_R_Yaw, L_R_Pitch, L_R_Roll


def difference_gaze_head(member, size, pin, num, eye='L', angle='Yaw', rotdir="", noise_flag=False,
                         noise_level=0.1):  # 读取用户特定日期和序号的视线数据和头部数据
    # ，以list返回视线和头部偏航角度之间的差异, num从1开始, eye='L' or 'R', angle='Yaw' or 'Pitch' or 'Roll'
    if eye not in ['L', 'R']:
        raise ValueError("eye must be 'L' or 'R'")
    if angle not in ['Yaw', 'Pitch', 'Roll']:
        raise ValueError("angle must be 'Yaw' or 'Pitch' or 'Roll'")
    # 数据存储在unity_processed_data目录下
    # member是studytype_user_date
    studytype = member.split('_')[0]
    user = member.split('_')[1]
    date = member.split('_')[2]

    data1 = pd.read_csv(os.path.join(rotdir,
                                     f"VRAuth{studytype[-1]}Angle/P{user}/GazeCalculate_data_{studytype}-{user}-{date}-{str(size)}-{str(pin)}-{str(num)}.csv"))
    data2 = pd.read_csv(os.path.join(rotdir,
                                     f"VRAuth{studytype[-1]}Angle/P{user}/Head_data_{studytype}-{user}-{date}-{str(size)}-{str(pin)}-{str(num)}.csv"))

    # 将读取的数据转换为DataFrame
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)

    # 从视线数据中计算初始偏航角度的偏移
    if noise_flag:
        df1[eye + '_' + angle] = add_noise(df1[eye + '_' + angle], noise_level=noise_level)
    gaze_zero = np.mean(df1[eye + '_' + angle][0:5])
    gaze_df = [x - gaze_zero if abs(x - gaze_zero) < 200 else (
        x - gaze_zero - 360 if x - gaze_zero > 200 else x - gaze_zero + 360) for x in df1[eye + '_' + angle]]

    # 从头部数据中计算初始偏航角度（Yaw）的偏移，并调整角度超过180度的情况
    if noise_flag:
        df2[angle] = add_noise(df2[angle], noise_level=noise_level)
    head_zero = np.mean([x if x < 180 else x - 360 for x in df2[angle][0:5]])
    head_df = [x - head_zero if x < 180 else x - head_zero - 360 for x in df2[angle]]

    # 返回视线和头部偏航角度之间的差异
    return [x - y for x, y in zip(gaze_df, head_df)]


def fourier_gaze(user, date, num, eye='L', angle='Yaw'):  # 读取用户特定日期和序号的视线数据，以list返回视线偏航角度的傅里叶变换结果
    if eye not in ['L', 'R']:
        raise ValueError("eye must be 'L' or 'R'")
    if angle not in ['Yaw', 'Pitch', 'Roll']:
        raise ValueError("angle must be 'Yaw' or 'Pitch' or 'Roll'")
    data1 = pd.read_csv(
        os.path.join("data", "GazeCalculate_data_" + user + "-" + date + "-" + str(num + 1) + "_unity_processed.csv"))
    df1 = pd.DataFrame(data1)

    gaze_zero = df1[eye + '_' + angle][0]
    gaze_df = [x - gaze_zero if abs(x - gaze_zero) < 200 else (
        x - gaze_zero - 360 if x - gaze_zero > 200 else x - gaze_zero + 360) for x in df1[eye + '_' + angle]]
    # gaze_df_slice = gaze_df[slice_l:slice_r]
    fft_gaze = np.fft.fft(gaze_df)
    freq_gaze = np.fft.fftfreq(len(gaze_df), 0.02)

    return gaze_df, fft_gaze, freq_gaze  # 返回视线偏航角度的傅里叶变换结果


def unity_quaternion_to_euler(x, y, z, w):  # result is different from Unity, idk why
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


def range_to_int_value(range_str):
    def range_to_int_start_end(range_str, value='start'):
        values = list(map(int, range_str.split('-')))
        return values[0] if value == 'start' else values[1]

    return range_to_int_start_end(range_str, 'end') - range_to_int_start_end(range_str, 'start')


def google_sheet_to_json(studytypes=['study1'], worksheet_name='simulation', credential_path="src/credentials.json", google_sheet_name="VRAuth被试招募",
                         json_save_path="src/data.json"): # *args is the tuple storing studytype
    
    import gspread
    def map_names_to_numbers(names): # 将人名按照首次出现的顺序映射成自然数列表
        name_to_number = {} # 首次出现的人名映射成1，第二次出现的人名映射成2，再次出现的人名不进入次dict
        number_list = [] # 用于存储映射后的自然数列表
        counter = 1
        for name in names:
            if name not in name_to_number:
                name_to_number[name] = counter
                counter += 1
            number_list.append(name_to_number[name])
        return number_list

    client = gspread.service_account(filename=credential_path)
    spreadsheet = client.open(google_sheet_name)
    sheet = spreadsheet.worksheet(worksheet_name) # worksheet是google sheet中的一个sheet
    # Fetch the first column values
    first_column = sheet.col_values(1)
    name_column = sheet.col_values(4)
    # for participant in range(count_study):
    first_occurrence = 2
    last_occurrence = len(name_column)

    # for i, value in enumerate(first_column, start=1):  # start=1 to start counting from row 1
    #     if is_string_in_tuple(value, studytypes):
    #         last_occurrence = i
    #         if first_occurrence is None:
    #             first_occurrence = i
    matching_rows = [i + 1 for i, cell_value in enumerate(first_column) if cell_value in studytypes]
    data_range = f"D{first_occurrence}:D{last_occurrence}"
    column_data = sheet.range(data_range)
    names = [cell.value for cell in column_data if cell.value.strip()]
    numbered_list = map_names_to_numbers(names)

    data_list = []

    for row in matching_rows:  # Adjust the range as needed
        # Generate or collect your data
        data_item = {
            "studytype": sheet.col_values(1)[row - 1],
            "names": numbered_list[row - 2],  # Indexing numbered_list correctly
            "date": sheet.col_values(2)[row - 1],
            "num_range": ""  # Added num_range attribute with empty value
        }
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


def head_eye_slice_quaternion_read(head_data_dir=None, eye_data_dir=None, segment_data_dir=None):
    data_head = pd.read_csv(head_data_dir)
    data_eye = pd.read_csv(eye_data_dir)
    # 切段文件是否存在, 不存在就用默认的切片方法
    ranges = None
    if not os.path.exists(segment_data_dir):
        warnings.warn(f"The file {segment_data_dir} does not exist.", Warning)
    else:
        with open(segment_data_dir, 'r') as file:
            text_data = file.read().strip()
            # Parse the ranges from the text data
            ranges = [list(map(int, r.split('-'))) for r in text_data.split(';') if r]

    return data_head, data_eye, ranges


# 1231update segment_data_dir为切断的文件路径
def feature_process_quaternion(data_head=None, data_eye=None, ranges=None,
                               noise_flag=False, noise_level=0.1):
    # 头的四元组
    QuaternionX_data = data_head['H-QuaternionX']
    if noise_flag:
        QuaternionX_data = add_noise(QuaternionX_data, noise_level)
    QuaternionX_data = QuaternionX_data - np.mean(QuaternionX_data[0:5])
    QuaternionX_data_smoothed = smooth_data(QuaternionX_data)
    d1 = np.array(QuaternionX_data_smoothed)
    d1_feat = extract_features(d1, ranges=ranges)
    QuaternionY_data = data_head['H-QuaternionY']
    if noise_flag:
        QuaternionY_data = add_noise(QuaternionY_data, noise_level)
    QuaternionY_data = QuaternionY_data - np.mean(QuaternionY_data[0:5])
    QuaternionY_data_smoothed = smooth_data(QuaternionY_data)
    d2 = np.array(QuaternionY_data_smoothed)
    d2_feat = extract_features(d2, ranges=ranges)
    QuaternionZ_data = data_head['H-QuaternionZ']
    if noise_flag:
        QuaternionZ_data = add_noise(QuaternionZ_data, noise_level)
    QuaternionZ_data = QuaternionZ_data - np.mean(QuaternionZ_data[0:5])
    QuaternionZ_data_smoothed = smooth_data(QuaternionZ_data)
    d3 = np.array(QuaternionZ_data_smoothed)
    d3_feat = extract_features(d3, ranges=ranges)
    QuaternionW_data = data_head['H-QuaternionW']
    if noise_flag:
        QuaternionW_data = add_noise(QuaternionW_data, noise_level)
    QuaternionW_data = QuaternionW_data - np.mean(QuaternionW_data[0:5])
    QuaternionW_data_smoothed = smooth_data(QuaternionW_data)
    d4 = np.array(QuaternionW_data_smoothed)
    d4_feat = extract_features(d4, ranges=ranges)
    # 头的坐标
    Vector3X_data = data_head['H-Vector3X']
    if noise_flag:
        Vector3X_data = add_noise(Vector3X_data, noise_level)
    Vector3X_data = Vector3X_data - np.mean(Vector3X_data[0:5])
    Vector3X_data_smoothed = smooth_data(Vector3X_data)
    v1 = np.array(Vector3X_data_smoothed)
    v1_feat = extract_features(v1, ranges=ranges)
    Vector3Y_data = data_head['H-Vector3Y']
    if noise_flag:
        Vector3Y_data = add_noise(Vector3Y_data, noise_level)
    Vector3Y_data = Vector3Y_data - np.mean(Vector3Y_data[0:5])
    Vector3Y_data_smoothed = smooth_data(Vector3Y_data)
    v2 = np.array(Vector3Y_data_smoothed)
    v2_feat = extract_features(v2, ranges=ranges)
    Vector3Z_data = data_head['H-Vector3Z']
    if noise_flag:
        Vector3Z_data = add_noise(Vector3Z_data, noise_level)
    Vector3Z_data = Vector3Z_data - np.mean(Vector3Z_data[0:5])
    Vector3Z_data_smoothed = smooth_data(Vector3Z_data)
    v3 = np.array(Vector3Z_data_smoothed)
    v3_feat = extract_features(v3, ranges=ranges)

    # 眼睛的四元组
    QuaternionX_data = data_eye['L-QuaternionX']
    if noise_flag:
        QuaternionX_data = add_noise(QuaternionX_data, noise_level)
    QuaternionX_data = QuaternionX_data - np.mean(QuaternionX_data[0:5])
    QuaternionX_data_smoothed = smooth_data(QuaternionX_data)
    d1_el = np.array(QuaternionX_data_smoothed)
    d1_el_feat = extract_features(d1_el, ranges=ranges)
    QuaternionY_data = data_eye['L-QuaternionY']
    if noise_flag:
        QuaternionY_data = add_noise(QuaternionY_data, noise_level)
    QuaternionY_data = QuaternionY_data - np.mean(QuaternionY_data[0:5])
    QuaternionY_data_smoothed = smooth_data(QuaternionY_data)
    d2_el = np.array(QuaternionY_data_smoothed)
    d2_el_feat = extract_features(d2_el, ranges=ranges)
    QuaternionZ_data = data_eye['L-QuaternionZ']
    if noise_flag:
        QuaternionZ_data = add_noise(QuaternionZ_data, noise_level)
    QuaternionZ_data = QuaternionZ_data - np.mean(QuaternionZ_data[0:5])
    QuaternionZ_data_smoothed = smooth_data(QuaternionZ_data)
    d3_el = np.array(QuaternionZ_data_smoothed)
    d3_el_feat = extract_features(d3_el, ranges=ranges)
    QuaternionW_data = data_eye['L-QuaternionW']
    if noise_flag:
        QuaternionW_data = add_noise(QuaternionW_data, noise_level)
    QuaternionW_data = QuaternionW_data - np.mean(QuaternionW_data[0:5])
    QuaternionW_data_smoothed = smooth_data(QuaternionW_data)
    d4_el = np.array(QuaternionW_data_smoothed)
    d4_el_feat = extract_features(d4_el, ranges=ranges)

    QuaternionX_data = data_eye['R-QuaternionX']
    if noise_flag:
        QuaternionX_data = add_noise(QuaternionX_data, noise_level)
    QuaternionX_data = QuaternionX_data - np.mean(QuaternionX_data[0:5])
    QuaternionX_data_smoothed = smooth_data(QuaternionX_data)
    d1_er = np.array(QuaternionX_data_smoothed)
    d1_er_feat = extract_features(d1_er, ranges=ranges)
    QuaternionY_data = data_eye['R-QuaternionY']
    if noise_flag:
        QuaternionY_data = add_noise(QuaternionY_data, noise_level)
    QuaternionY_data = QuaternionY_data - np.mean(QuaternionY_data[0:5])
    QuaternionY_data_smoothed = smooth_data(QuaternionY_data)
    d2_er = np.array(QuaternionY_data_smoothed)
    d2_er_feat = extract_features(d2_er, ranges=ranges)
    QuaternionZ_data = data_eye['R-QuaternionZ']
    if noise_flag:
        QuaternionZ_data = add_noise(QuaternionZ_data, noise_level)
    QuaternionZ_data = QuaternionZ_data - np.mean(QuaternionZ_data[0:5])
    QuaternionZ_data_smoothed = smooth_data(QuaternionZ_data)
    d3_er = np.array(QuaternionZ_data_smoothed)
    d3_er_feat = extract_features(d3_er, ranges=ranges)
    QuaternionW_data = data_eye['R-QuaternionW']
    if noise_flag:
        QuaternionW_data = add_noise(QuaternionW_data, noise_level)
    QuaternionW_data = QuaternionW_data - np.mean(QuaternionW_data[0:5])
    QuaternionW_data_smoothed = smooth_data(QuaternionW_data)
    d4_er = np.array(QuaternionW_data_smoothed)
    d4_er_feat = extract_features(d4_er, ranges=ranges)

    return d1, d1_feat, d2, d2_feat, d3, d3_feat, d4, d4_feat, v1, v1_feat, v2, v2_feat, v3, v3_feat, d1_el, d1_el_feat, \
        d2_el, d2_el_feat, d3_el, d3_el_feat, d4_el, d4_el_feat, d1_er, d1_er_feat, d2_er, d2_er_feat, d3_er, d3_er_feat, d4_er, d4_er_feat


def feature_process_angle(eye_data_dir=None, head_data_dir=None, noise_flag=False, noise_level=0.1):
    data_head = pd.read_csv(head_data_dir)
    Yaw_data = data_head['Yaw']
    if noise_flag:
        Yaw_data = add_noise(Yaw_data, noise_level)
    Yaw_data = Yaw_data - np.mean(Yaw_data[0:5])
    Yaw_data_smoothed = smooth_data(Yaw_data)
    d1 = np.array(Yaw_data_smoothed)
    d1_feat = extract_features(d1)
    Pitch_data = data_head['Pitch']
    if noise_flag:
        Pitch_data = add_noise(Pitch_data, noise_level)
    Pitch_data = Pitch_data - np.mean(Pitch_data[0:5])
    Pitch_data_smoothed = smooth_data(Pitch_data)
    d2 = np.array(Pitch_data_smoothed)
    d2_feat = extract_features(d2)
    Roll_data = data_head['Roll']
    if noise_flag:
        Roll_data = add_noise(Roll_data, noise_level)
    Roll_data = Roll_data - np.mean(Roll_data[0:5])
    Roll_data_smoothed = smooth_data(Roll_data)
    d3 = np.array(Roll_data_smoothed)
    d3_feat = extract_features(d3)

    # Eye points
    data_eye = pd.read_csv(eye_data_dir)
    Yaw_data = data_eye['L_Yaw']
    if noise_flag:
        Yaw_data = add_noise(Yaw_data, noise_level)
    Yaw_data = Yaw_data - np.mean(Yaw_data[0:5])
    Yaw_data_smoothed = smooth_data(Yaw_data)
    d1_el = np.array(Yaw_data_smoothed)
    d1_el_feat = extract_features(d1_el)
    Pitch_data = data_eye['L_Pitch']
    if noise_flag:
        Pitch_data = add_noise(Pitch_data, noise_level)
    Pitch_data = Pitch_data - np.mean(Pitch_data[0:5])
    Pitch_data_smoothed = smooth_data(Pitch_data)
    d2_el = np.array(Pitch_data_smoothed)
    d2_el_feat = extract_features(d2_el)
    Roll_data = data_eye['L_Roll']
    if noise_flag:
        Roll_data = add_noise(Roll_data, noise_level)
    Roll_data = Roll_data - np.mean(Roll_data[0:5])
    Roll_data_smoothed = smooth_data(Roll_data)
    d3_el = np.array(Roll_data_smoothed)
    d3_el_feat = extract_features(d3_el)

    # 右眼
    Yaw_data = data_eye['R_Yaw']
    if noise_flag:
        Yaw_data = add_noise(Yaw_data, noise_level)
    Yaw_data = Yaw_data - np.mean(Yaw_data[0:5])
    Yaw_data_smoothed = smooth_data(Yaw_data)
    d1_er = np.array(Yaw_data_smoothed)
    d1_er_feat = extract_features(d1_er)
    Pitch_data = data_eye['R_Pitch']
    if noise_flag:
        Pitch_data = add_noise(Pitch_data, noise_level)
    Pitch_data = Pitch_data - np.mean(Pitch_data[0:5])
    Pitch_data_smoothed = smooth_data(Pitch_data)
    d2_er = np.array(Pitch_data_smoothed)
    d2_er_feat = extract_features(d2_er)
    Roll_data = data_eye['R_Roll']
    if noise_flag:
        Roll_data = add_noise(Roll_data, noise_level)
    Roll_data = Roll_data - np.mean(Roll_data[0:5])
    Roll_data_smoothed = smooth_data(Roll_data)
    d3_er = np.array(Roll_data_smoothed)
    d3_er_feat = extract_features(d3_er)

    return d1, d1_feat, d2, d2_feat, d3, d3_feat, d1_el, d1_el_feat, \
        d2_el, d2_el_feat, d3_el, d3_el_feat, d1_er, d1_er_feat, d2_er, d2_er_feat, d3_er, d3_er_feat


def merged_array_generator(data_head, data_eye, ranges, member, size, pin, num, model, rotdir, noise_flag=None,
                           noise_level=0.1):  # num从1开始

    # 四元组 calculate为世界坐标，raw为头部局域坐标下的旋转数值
    d1, d1_feat, d2, d2_feat, d3, d3_feat, d4, d4_feat, v1, v1_feat, v2, v2_feat, v3, v3_feat, d1_el, d1_el_feat, d2_el, \
        d2_el_feat, d3_el, d3_el_feat, d4_el, d4_el_feat, d1_er, d1_er_feat, d2_er, d2_er_feat, d3_er, d3_er_feat, d4_er, \
        d4_er_feat = feature_process_quaternion(data_head=data_head, data_eye=data_eye, ranges=ranges,
                                                noise_flag=noise_flag, noise_level=noise_level)

    if model == 'head':
        merged_array = np.concatenate(
            [d1_feat, d2_feat, d3_feat, d4_feat, v1_feat, v2_feat, v3_feat])
        # [d1_feat, d2_feat, d3_feat])

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
        diff_yaw_data = difference_gaze_head(member, size, pin, num, rotdir=rotdir, noise_flag=noise_flag,
                                             noise_level=noise_level)
        diff_yaw_smooth = smooth_data(diff_yaw_data, window_parameter=9)
        dy_el_feat = extract_features(np.array(diff_yaw_smooth))
        diff_pitch_data = difference_gaze_head(member, size, pin, num, eye='L', angle='Pitch', rotdir=rotdir,
                                               noise_flag=noise_flag, noise_level=noise_level)
        diff_pitch_smooth = smooth_data(diff_pitch_data, window_parameter=9)
        dp_el_feat = extract_features(np.array(diff_pitch_smooth))
        diff_roll_data = difference_gaze_head(member, size, pin, num, eye='L', angle='Roll', rotdir=rotdir,
                                              noise_flag=noise_flag, noise_level=noise_level)
        diff_roll_smooth = smooth_data(diff_roll_data, window_parameter=9)
        dr_el_feat = extract_features(np.array(diff_roll_smooth))
        # 利用特征：切10段的特征
        merged_array = np.concatenate([dy_el_feat, dp_el_feat, dr_el_feat]
                                      )

    else:
        diff_yaw_data = difference_gaze_head(member, size, pin, num, rotdir=rotdir, noise_flag=noise_flag,
                                             noise_level=noise_level)
        diff_yaw_smooth = smooth_data(diff_yaw_data, window_parameter=9)
        dy_el_feat = extract_features(np.array(diff_yaw_smooth))
        diff_pitch_data = difference_gaze_head(member, size, pin, num, eye='L', angle='Pitch', rotdir=rotdir,
                                               noise_flag=noise_flag, noise_level=noise_level)
        diff_pitch_smooth = smooth_data(diff_pitch_data, window_parameter=9)
        dp_el_feat = extract_features(np.array(diff_pitch_smooth))
        diff_roll_data = difference_gaze_head(member, size, pin, num, eye='L', angle='Roll', rotdir=rotdir,
                                              noise_flag=noise_flag, noise_level=noise_level)
        diff_roll_smooth = smooth_data(diff_roll_data, window_parameter=9)
        dr_el_feat = extract_features(np.array(diff_roll_smooth))
        merged_array = np.concatenate(
            [d1_feat, d2_feat, d3_feat, d4_feat, v1_feat, v2_feat, v3_feat, d1_el_feat, d2_el_feat,
             d3_el_feat,
             d4_el_feat, d1_er_feat, d2_er_feat, d3_er_feat, d4_er_feat,
             dy_el_feat, dp_el_feat, dr_el_feat])

    # print(d1)
    # if np.isnan(d1_feat).any():
    #     print("NaN values found in d1_feat")
    #     # 定位NaN值
    #     print(np.argwhere(np.isnan(d1_feat)))

    return merged_array


# 返回整理好的能用于模型训练测试的X和Y
def data_augment_and_label(studytype_users_dates_range, rotdir=None, model="", size_list=None,
                           pin_list=None, default_authentications_per_person=6,
                           positive_label=None, noise_level=0.1, augment_time=1):  # 返回scaled后的原始数据和标签，scaled后的增强后的数据和标签

    studytype_user_date_size_pin_num_pair = []  # studytype, user, date, size, pin, num 的所有排列组合，用于数据增强时循环增强所有正标签里的数据
    result_array = np.array([])
    print(f"function data_augment_and_label start")
    print(f"studytype_users_dates_range: {studytype_users_dates_range}")
    studytype = studytype_users_dates_range[0].split('_')[0]  # studytype只有一种
    labels = []
    binary_labels = []

    for member in studytype_users_dates_range:
        user = member.split('_')[1]
        date = member.split('_')[2]
        num_range = member.split('_')[3]
        range_start = int(num_range.split('-')[0]) if num_range else 1
        range_end = int(num_range.split('-')[1]) if num_range else default_authentications_per_person

        # 全部可能的组合
        studytype_user_date_size_pin_num_pair.extend([x for x in
                                                      itertools.product([studytype], [user], [date], size_list,
                                                                        pin_list, range(range_start, range_end + 1))])

        # 标签的生成，按照人名的唯一性
        # labels.extend(np.repeat(user, len(size_list) * len(pin_list) * (range_end - range_start + 1) * augment_time))
        # binary_labels.extend([1 if user in positive_label else 0 for _ in
        #                       range(len(size_list) * len(pin_list) * (range_end - range_start + 1) * augment_time)])

        # 特征拼接，数据增强
        for size in size_list:
            for pin in pin_list:
                for num in range(range_start, range_end + 1):
                    # 文件读取
                    head_path = rotdir + f"VRAuth{studytype[-1]}/P{user}/Head_data_{studytype}-{user}-{date}-{str(size)}-{str(pin)}-{str(num)}.csv"
                    eye_path = rotdir + f"VRAuth{studytype[-1]}/P{user}/GazeCalculate_data_{studytype}-{user}-{date}-{str(size)}-{str(pin)}-{str(num)}.csv"
                    segment_path = rotdir + f"VRAuth{studytype[-1]}/P{user}/Saccades_{studytype}-{user}-{date}-{str(size)}-{str(pin)}-{str(num)}.txt"
                    data_head, data_eye, ranges = head_eye_slice_quaternion_read(head_data_dir=head_path,
                                                                                 eye_data_dir=eye_path,
                                                                                 segment_data_dir=segment_path)
                    # 1.1 update 数据增强
                    for i in range(1, augment_time + 1):
                        noise_flag = False if i == 1 else True  # 增强倍数大于1则选择增强，首次循环（i=1）为False

                        # 对于该增强噪声的水平 返回该member, size, pin, num的特征, 返回一维X向量
                        try: # 跳过saccade文件里只有一段fixation的情况，
                            merged_array = merged_array_generator(data_head=data_head, data_eye=data_eye, ranges=ranges,
                                                                member=member, rotdir=rotdir, model=model, size=size, pin=pin,
                                                                num=num, noise_flag=noise_flag, noise_level=noise_level)
                        except IndexError as e:
                            print(f"member: {member}, size: {size}, pin: {pin}, num: {num}, augment_time: {i}")
                        if not np.isnan(merged_array).any():
                            labels.append(user) # label生成，如果
                            binary_labels.append(1 if user in positive_label else 0) # 标签的生成，按照人名的唯一性
                            # 将所有特征堆叠起来，每一行是一个特征
                            result_array = np.vstack([result_array, merged_array]) if result_array.size else merged_array

    scaled_data = result_array

    # 识别正类样本
    positive_indices = np.where(binary_labels == 1)[0]

    # 确定增强的正样本数量以达到大约50%的正样本比例
    total_samples_needed = len(binary_labels)  # 总样本数
    positive_samples_needed = int(total_samples_needed) - 2 * len(positive_indices)  # 需要增强的正样本数

    # 如果需要增强的样本数为负数或零，则不执行任何操作
    if positive_samples_needed > 0:
        # 选择正样本进行复制和添加噪声
        users_to_copy = np.random.choice(positive_label, size=positive_samples_needed, replace=True)
        loop_num = 0
        index = 0
        positive_features_to_augment = np.array([])
        binary_labels_to_concatenate = []
        # studytype user date size pin num

        while loop_num < positive_samples_needed:
            user_to_copy = users_to_copy[loop_num]
            studytype_user_date_size_pin_num_pair_to_copy = [x for x in studytype_user_date_size_pin_num_pair if
                                                             x[1] == user_to_copy]  # user_to_copy的所有组合
            studytype_to_copy = studytype_user_date_size_pin_num_pair_to_copy[index][0]
            date_to_copy = studytype_user_date_size_pin_num_pair_to_copy[index][2]
            size_to_copy = studytype_user_date_size_pin_num_pair_to_copy[index][3]
            pin_to_copy = studytype_user_date_size_pin_num_pair_to_copy[index][4]
            num_to_copy = studytype_user_date_size_pin_num_pair_to_copy[index][5]

            member_to_copy = f"{studytype_to_copy}_{user_to_copy}_{date_to_copy}"  # 用于merged_array_generator的member参数
            head_path = rotdir + f"VRAuth{studytype_to_copy[-1]}/P{user_to_copy}/Head_data_{studytype_to_copy}-{user_to_copy}-{date_to_copy}-{str(size_to_copy)}-{str(pin_to_copy)}-{str(num_to_copy)}.csv"
            eye_path = rotdir + f"VRAuth{studytype_to_copy[-1]}/P{user_to_copy}/GazeCalculate_data_{studytype_to_copy}-{user_to_copy}-{date_to_copy}-{str(size_to_copy)}-{str(pin_to_copy)}-{str(num_to_copy)}.csv"
            segment_path = rotdir + f"VRAuth{studytype_to_copy[-1]}/P{user_to_copy}/Saccades_{studytype_to_copy}-{user_to_copy}-{date_to_copy}-{str(size_to_copy)}-{str(pin_to_copy)}-{str(num_to_copy)}.txt"
            data_head, data_eye, ranges = head_eye_slice_quaternion_read(head_data_dir=head_path,
                                                                         eye_data_dir=eye_path,
                                                                         segment_data_dir=segment_path)

            try: # 跳过saccade文件里只有一段fixation的情况，
                merged_array_augmented = merged_array_generator(data_head=data_head, data_eye=data_eye, ranges=ranges,
                                                            member=member_to_copy, rotdir=rotdir, model=model,
                                                            size=size_to_copy, pin=pin_to_copy, num=num_to_copy,
                                                            noise_flag=True, noise_level=noise_level)
            except IndexError as e:
                            print(f"member: {member}, size: {size}, pin: {pin}, num: {num}, augment_time: {i}")
            
            if not np.isnan(merged_array_augmented).any(): # 如果增强后出现空特征
                binary_labels_to_concatenate.append(1)
                positive_features_to_augment = np.vstack([positive_features_to_augment,
                                                      merged_array_augmented]) if positive_features_to_augment.size else merged_array_augmented

            index = (index + 1) % len(studytype_user_date_size_pin_num_pair_to_copy)

            loop_num += 1

        # 将增强的样本合并回原始数据集
        result_array_augmented = np.concatenate((result_array, positive_features_to_augment), axis=0)
        print(f"result_array.shape: {result_array.shape}")
        print(f"positive_features_to_augment.shape: {positive_features_to_augment.shape}")
        print(f"result_array_augmented.shape: {result_array_augmented.shape}")
        # label_augmented = np.concatenate((labels, labels[indices_to_copy]), axis=0)
        binary_labels_augmented = np.concatenate((binary_labels, binary_labels_to_concatenate), axis=0)
        scaled_data_augmented = result_array_augmented
    else:
        # 如果不需要增加正样本，则保持原始数据不变
        scaled_data_augmented = scaled_data
        # label_augmented = labels
        binary_labels_augmented = binary_labels

    return scaled_data, np.array(labels), np.array(binary_labels), scaled_data_augmented, np.array(
        binary_labels_augmented)

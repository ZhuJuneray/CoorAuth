"""
1.change quaternions to euler angles
2.draw the euler angles diagram
3.save the diagram
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os
from cycler import cycler
import numpy as np

plt.rcParams['axes.prop_cycle'] = cycler(color=plt.cm.tab20.colors)
# 
# ---need to modify regarding your csv file name---
user_names=['yf','zjr']
dates=['1111']
repeat_num=2
plot_row1=2; plot_column1=4
plot_row2=2; plot_column2=4
fig_1, axs_1 = plt.subplots(plot_row1, plot_column1, figsize=(20, 10))
fig_2, axs_2 = plt.subplots(plot_row2, plot_column2, figsize=(20, 10))
#---modify end---

# def smoothed_outlier()
os.chdir(os.path.join(os.getcwd(),'VRAuth'))


def replace_local_outliers(arr, window_size=5, threshold=1.5):
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
def smooth_data(arr, window_parameter=31, polyorder_parameter=2):
    arr_smoothed = savgol_filter(arr, window_length=window_parameter, polyorder=polyorder_parameter)
    return arr_smoothed
# Using the same logic as the Unity example, but implemented in Python
# Assuming the quaternion is in the format (w, x, y, z)

def unity_quaternion_to_euler(x, y, z, w):
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

def quaternion_to_euler (x, y, z, w):
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


def quaternion_to_euler_df(dataframe):
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

# Example usage:
# Assuming `data1` is your DataFrame
# euler_angles_df = quaternion_to_euler(data1)


for user in user_names:
    for num in range(repeat_num):
        for date in dates:
            execute_block = True # skip specific line drawing: True for skipping and False for drawing all line
            if execute_block:
                if (date=="1108" or date=="1109") and (user=='jyc' or user=='lhy' or user=='wgh' or user=='yf'):
                # if user=='zjr':
                    continue 

            data1=pd.read_csv("GazeRaw_data_"+user+'-'+date+'-'+str(num+1)+'.csv')
            data2=pd.read_csv("GazeCalculate_data_"+user+'-'+date+'-'+str(num+1)+'.csv')
            # print(data.index)
            
            

            L_QuaternionX_raw_data=data1['L-QuaternionX']
            L_QuaternionY_raw_data=data1['L-QuaternionY']
            L_QuaternionZ_raw_data=data1['L-QuaternionZ']
            L_QuaternionW_raw_data=data1['L-QuaternionW']
            R_QuaternionX_raw_data=data1['R-QuaternionX']
            R_QuaternionY_raw_data=data1['R-QuaternionY']
            R_QuaternionZ_raw_data=data1['R-QuaternionZ']
            R_QuaternionW_raw_data=data1['R-QuaternionW']

            # axs_1[0, 0].plot(L_QuaternionX_raw_data,label=user+str(num+1)+'_'+date)
            # axs_1[0, 0].set_title('L_QuaternionX_raw_data')
            # axs_1[0, 1].plot(L_QuaternionY_raw_data,label=user+str(num+1)+'_'+date)
            # axs_1[0, 1].set_title('L_QuaternionY_raw_data')
            # axs_1[0, 2].plot(L_QuaternionZ_raw_data,label=user+str(num+1)+'_'+date)
            # axs_1[0, 2].set_title('L_QuaternionZ_raw_data')
            # axs_1[0, 3].plot(L_QuaternionW_raw_data,label=user+str(num+1)+'_'+date)
            # axs_1[0, 3].set_title('L_QuaternionW_raw_data')
            # axs_1[1, 0].plot(R_QuaternionX_raw_data,label=user+str(num+1)+'_'+date)
            # axs_1[1, 0].set_title('R_QuaternionX_raw_data')
            # axs_1[1, 1].plot(R_QuaternionY_raw_data,label=user+str(num+1)+'_'+date)
            # axs_1[1, 1].set_title('R_QuaternionY_raw_data')
            # axs_1[1, 2].plot(R_QuaternionZ_raw_data,label=user+str(num+1)+'_'+date)
            # axs_1[1, 2].set_title('R_QuaternionZ_raw_data')
            # axs_1[1, 3].plot(R_QuaternionW_raw_data,label=user+str(num+1)+'_'+date)
            # axs_1[1, 3].set_title('R_QuaternionW_raw_data')

            L_euler_angles_df = pd.DataFrame(columns=['Yaw', 'Pitch', 'Roll'])
            R_euler_angles_df = pd.DataFrame(columns=['Yaw', 'Pitch', 'Roll'])

            L_QuaternionX_calculated_data=data2['L-QuaternionX']
            L_QuaternionY_calculated_data=data2['L-QuaternionY']
            L_QuaternionZ_calculated_data=data2['L-QuaternionZ']
            L_QuaternionW_calculated_data=data2['L-QuaternionW']
            for x, y, z, w in zip(data2['L-QuaternionX'], data2['L-QuaternionY'], data2['L-QuaternionZ'], data2['L-QuaternionW']):
                L_euler_angles = unity_quaternion_to_euler(x, y, z, w)
                L_euler_angles_df = L_euler_angles_df.append({'Yaw': L_euler_angles[0], 'Pitch': L_euler_angles[1], 'Roll': L_euler_angles[2]}, ignore_index=True)

            for x, y, z, w in zip(data2['L-QuaternionX'], data2['L-QuaternionY'], data2['L-QuaternionZ'], data2['L-QuaternionW']):
                R_euler_angles = unity_quaternion_to_euler(x, y, z, w)
                R_euler_angles_df = R_euler_angles_df.append({'Yaw': R_euler_angles[0], 'Pitch': R_euler_angles[1], 'Roll': R_euler_angles[2]}, ignore_index=True)
            
            R_QuaternionX_calculated_data=data2['R-QuaternionX']
            R_QuaternionY_calculated_data=data2['R-QuaternionY']
            R_QuaternionZ_calculated_data=data2['R-QuaternionZ']
            R_QuaternionW_calculated_data=data2['R-QuaternionW']

            axs_2[0, 0].plot(L_euler_angles_df['Yaw'],label=user+str(num+1)+'_'+date)
            axs_2[0, 0].set_title('L_Yaw')
            axs_2[0, 1].plot(L_euler_angles_df['Pitch'],label=user+str(num+1)+'_'+date)
            axs_2[0, 1].set_title('L_Pitch')
            axs_2[0, 2].plot(L_euler_angles_df['Roll'],label=user+str(num+1)+'_'+date)
            axs_2[0, 2].set_title('L_Roll')

            axs_2[1, 0].plot(R_euler_angles_df['Yaw'],label=user+str(num+1)+'_'+date)
            axs_2[1, 0].set_title('R_Yaw')
            axs_2[1, 1].plot(R_euler_angles_df['Pitch'],label=user+str(num+1)+'_'+date)
            axs_2[1, 1].set_title('R_Pitch')
            axs_2[1, 2].plot(R_euler_angles_df['Roll'],label=user+str(num+1)+'_'+date)
            axs_2[1, 2].set_title('R_Roll')

            # axs_2[0, 0].plot(L_QuaternionX_calculated_data,label=user+str(num+1)+'_'+date)
            # axs_2[0, 0].set_title('L_QuaternionX_calculated_data')
            # axs_2[0, 1].plot(L_QuaternionY_calculated_data,label=user+str(num+1)+'_'+date)
            # axs_2[0, 1].set_title('L_QuaternionY_calculated_data')
            # axs_2[0, 2].plot(L_QuaternionZ_calculated_data,label=user+str(num+1)+'_'+date)
            # axs_2[0, 2].set_title('L_QuaternionZ_calculated_data')
            # axs_2[0, 3].plot(L_QuaternionW_calculated_data,label=user+str(num+1)+'_'+date)
            # axs_2[0, 3].set_title('L_QuaternionW_calculated_data')
            # axs_2[1, 0].plot(R_QuaternionX_calculated_data,label=user+str(num+1)+'_'+date)
            # axs_2[1, 0].set_title('R_QuaternionX_calculated_data')
            # axs_2[1, 1].plot(R_QuaternionY_calculated_data,label=user+str(num+1)+'_'+date)
            # axs_2[1, 1].set_title('R_QuaternionY_calculated_data')
            # axs_2[1, 2].plot(R_QuaternionZ_calculated_data,label=user+str(num+1)+'_'+date)
            # axs_2[1, 2].set_title('R_QuaternionZ_calculated_data')
            # axs_2[1, 3].plot(R_QuaternionW_calculated_data,label=user+str(num+1)+'_'+date)
            # axs_2[1, 3].set_title('R_QuaternionW_calculated_data')

            axs_1[0, 0].legend()
            # for i in range(3):
            #     for j in range(4):
            #         axs_2[i, j].legend()
            axs_2[0, 0].legend()

plt.tight_layout()
fig_1.savefig('Gaze_data_raw' + str(user_names) + '.png')
fig_2.savefig('Gaze_data_calculated' + str(user_names) + '.png')
# plt.show()

subfolder = "single_figures"
if not os.path.exists(subfolder):
    os.makedirs(subfolder)
for axs in (axs_1,axs_2):
    for i in range(plot_row1):
        for j in range(plot_column1):
            # 提取单个子图
            ax = axs[i, j]

            fig_new = plt.figure()
            ax_new = fig_new.add_subplot(111)

            # 复制原始子图的内容到新的ax中
            ax_new.set_title(axs[i,j].get_title())
            ax_new.set_label (axs_2[0, 0].get_label())
            ax_new.legend()
            for line in ax.get_lines():
                ax_new.plot(line.get_xdata(), line.get_ydata(), color=line.get_color())
            

            # 保存单个子图
            plt.savefig(os.path.join(subfolder , axs[i,j].get_title()+ ".png"), bbox_inches='tight', dpi=300)



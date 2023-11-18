"""
Author: June Ray
Email: juneray2003@gmail.com
GitHub: https://github.com/ZhuJunray/
Version: 1.0.0
Date: 2023-11-08
Description: This is a Python script to draw csv data from Quest.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os
from cycler import cycler

plt.rcParams['axes.prop_cycle'] = cycler(color=plt.cm.tab20.colors)

# ---need to modify regarding your csv file name---
user_names=["zjr",'zs','yf','wgh','lhy','jyc']
dates=['1111']
repeat_num=2
plot_row=3; plot_column=3
fig_1, axs_1 = plt.subplots(plot_row, plot_column, figsize=(20, 10))
fig_2, axs_2 = plt.subplots(3, 4, figsize=(20, 10))
#---modify end---

# def smoothed_outlier()

import numpy as np

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

for user in user_names:
    for num in range(repeat_num):
        for date in dates:
            execute_block = True # skip specific line drawing: True for skipping and False for drawing all line
            if execute_block:
                if (date=="1108" or date=="1109") and (user=='jyc' or user=='lhy' or user=='wgh' or user=='yf'):
                # if user=='zjr':
                    continue 

            data=pd.read_csv("Head_data_"+user+'-'+date+'-'+str(num+1)+'.csv')
            # print(data.index)
            
            

            Vector3X_data=data['H-Vector3X']
            Vector3X_data_smoothed = smooth_data(Vector3X_data)
            # Vector3X_data_smoothed = savgol_filter(data['H-Vector3X'], window_length=31, polyorder=2)
            Vector3X_data_smoothed_outlier = replace_local_outliers(Vector3X_data_smoothed,window_size=2,threshold=1.5)
            Vector3X_data_smoothed_slope = np.diff(Vector3X_data_smoothed)


            Vector3Y_data=data['H-Vector3Y']
            Vector3Y_data_smoothed = smooth_data(Vector3Y_data)
            # Vector3Y_data_smoothed = savgol_filter(data['H-Vector3Y'], window_length=31, polyorder=10)
            Vector3Y_data_smoothed_slope = np.diff(Vector3Y_data_smoothed)

            Vector3Z_data=data['H-Vector3Z']
            Vector3Z_data_smoothed = smooth_data(Vector3Z_data)
            # Vector3Z_data_smoothed_slope = np.diff(Vector3X_data_smoothed)
            # print(Vector3Z_data_smoothed)
            Vector3Z_data_smoothed_slope = np.diff(Vector3Z_data_smoothed)

            QuaternionX_data=data['H-QuaternionX']
            QuaternionX_data_smoothed = smooth_data(QuaternionX_data)
            QuaternionX_data_smoothed_slope = np.diff(QuaternionX_data_smoothed)

            QuaternionY_data=data['H-QuaternionY']
            QuaternionY_data_smoothed = smooth_data(QuaternionY_data)
            QuaternionY_data_smoothed_slope = np.diff(QuaternionY_data_smoothed)

            QuaternionZ_data=data['H-QuaternionZ']
            QuaternionZ_data_smoothed = smooth_data(QuaternionZ_data)
            QuaternionZ_data_smoothed_slope = np.diff(QuaternionZ_data_smoothed)

            QuaternionW_data=data['H-QuaternionW']
            QuaternionW_data_smoothed = smooth_data(QuaternionW_data)
            QuaternionW_data_smoothed_slope = np.diff(QuaternionW_data_smoothed)

            axs_1[0, 0].plot(Vector3X_data-Vector3X_data[0],label=user+str(num+1)+'_'+date)
            axs_1[0, 0].set_title('Vector3X')
            axs_1[1, 0].plot(Vector3X_data_smoothed-Vector3X_data_smoothed[0])
            axs_1[1, 0].set_title('Vector3X_smoothed')
            axs_1[2, 0].plot(Vector3X_data_smoothed_slope)
            axs_1[2, 0].set_title('Vector3X_smoothed_slope')
            # axs_1[2, 0].plot(Vector3X_data_smoothed_outlier)
            # axs_1[2, 0].set_title('Vector3X_smoothed_outlier')

            axs_1[0, 1].plot(Vector3Y_data-Vector3Y_data[0])
            axs_1[0, 1].set_title('Vector3Y_data')
            axs_1[1, 1].plot(Vector3Y_data_smoothed-Vector3Y_data_smoothed[0])
            axs_1[1, 1].set_title('Vector3Y_smoothed')
            axs_1[2, 1].plot(Vector3Y_data_smoothed_slope)
            axs_1[2, 1].set_title('Vector3Y_smoothed_slope')
            # axs_1[1, 0].plot(Vector3Z_data-Vector3Z_data[0])
            # axs_1[1, 0].set_title('Vector3Z')

            axs_1[0, 2].plot(Vector3Z_data-Vector3Z_data[0])
            axs_1[0, 2].set_title('Vector3Z_data')
            axs_1[1, 2].plot(Vector3Z_data_smoothed-Vector3Z_data_smoothed[0])
            axs_1[1, 2].set_title('Vector3Z_smoothed')
            axs_1[2, 2].plot(Vector3Z_data_smoothed_slope)
            axs_1[2, 2].set_title('Vector3Z_smoothed_slope')

            # axs_1[2, 0].plot(Vector3Z_data_smoothed-Vector3Z_data_smoothed[0],label=user+str(num+1)+'_'+date)
            # axs_1[2, 0].set_title('Vector3Z_smoothed')

            axs_2[0, 0].plot(QuaternionX_data-QuaternionX_data[0],label=user+str(num+1)+'_'+date)
            axs_2[0, 0].set_title('QuaternionX')
            axs_2[1, 0].plot(QuaternionX_data_smoothed-QuaternionX_data_smoothed[0])
            axs_2[1, 0].set_title('QuaternionX_smoothed')
            axs_2[2, 0].plot(QuaternionX_data_smoothed_slope)
            axs_2[2, 0].set_title('QuaternionX_smoothed_slope')
            axs_2[0, 1].plot(QuaternionY_data-QuaternionY_data[0])
            axs_2[0, 1].set_title('QuaternionY')
            axs_2[1, 1].plot(QuaternionY_data_smoothed-QuaternionY_data_smoothed[0])
            axs_2[1, 1].set_title('QuaternionY_smoothed')
            axs_2[2, 1].plot(QuaternionY_data_smoothed_slope)
            axs_2[2, 1].set_title('QuaternionY_smoothed_slope')
            axs_2[0, 2].plot(QuaternionZ_data-QuaternionZ_data[0])
            axs_2[0, 2].set_title('QuaternionZ')
            axs_2[1, 2].plot(QuaternionZ_data_smoothed-QuaternionZ_data_smoothed[0])
            axs_2[1, 2].set_title('QuaternionZ_smoothed')
            axs_2[2, 2].plot(QuaternionZ_data_smoothed_slope)
            axs_2[2, 2].set_title('QuaternionZ_smoothed_slope')
            axs_2[0, 3].plot(QuaternionW_data-QuaternionW_data[0])
            axs_2[0, 3].set_title('QuaternionW')
            axs_2[1, 3].plot(QuaternionW_data_smoothed-QuaternionW_data_smoothed[0])
            axs_2[1, 3].set_title('QuaternionW_smoothed')
            axs_2[2, 3].plot(QuaternionW_data_smoothed_slope)
            axs_2[2, 3].set_title('QuaternionW_smoothed_slope')

            # axs_1[1, 0].plot(Vector3Z_data_smoothed_slope,label=user+str(num+1)+'_'+date)
            # axs_1[1, 0].set_title('Vector3Z_smoothed_slope')
            for i in range(plot_row):
                for j in range(plot_column):
                    axs_1[i, j].legend()
            
            for i in range(3):
                for j in range(4):
                    axs_2[i, j].legend()

plt.tight_layout()
fig_1.savefig('Head_data_drawed_xyz.png')
fig_2.savefig('Head_data_drawed_quaternionXYZW.png')
# plt.show()

subfolder = "single_figures"
if not os.path.exists(subfolder):
    os.makedirs(subfolder)
for axs in (axs_1,axs_2):
    for i in range(plot_row):
        for j in range(plot_column):
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



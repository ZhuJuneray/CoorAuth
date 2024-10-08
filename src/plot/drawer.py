import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from scipy.signal import savgol_filter
from cycler import cycler
import os
from data_preprocess import replace_local_outliers
from data_preprocess import smooth_data
from data_preprocess import extract_features
from data_preprocess import fourier_gaze
from data_preprocess import difference_gaze_lr_euler_angle
from data_preprocess import difference_gaze_head
import re, json
import itertools
from data_preprocess import read_data_name_from_json
from collections import defaultdict
from scipy import stats
import pingouin as pg
from difflib import SequenceMatcher
import warnings, sys
import seaborn as sns
from matplotlib.ticker import MaxNLocator
plt.rcParams.update({'font.size': 20})

# os.chdir(os.path.join(os.getcwd(),'data'))

# ---need to modify regarding your csv file name---
user_names = ["zjr", "zs"]
dates = "1102"
repeat_num = 3


#---modify end---
def expression_data_drawer(user_names, date, repeat_num):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    for user in user_names:
        for num in range(repeat_num):

            execute_block = False # skip specific line drawing: True for skipping and False for drawing all line
            if execute_block:
                if num==2 and user=='zjr':
                    continue 

            data=pd.read_csv("expression_data_"+user+date+"_"+str(num+1)+'.csv',header=None,skiprows=1)
            data=data.drop(columns=[0])
            print(data.index)
            NoseWrinklerR_data=data.iloc[0]
            CheekRaiserR_data=data.iloc[1]
            LidTightenerR_data=data.iloc[2]
            UpperLipRaiserR_data=data.iloc[3]
            axs[0, 0].plot(NoseWrinklerR_data,label=user+str(num+1))
            axs[0, 0].set_title('NoseWrinklerR')
            axs[0, 1].plot(CheekRaiserR_data,label=user+str(num+1))
            axs[0, 1].set_title('CheekRaiserR')
            axs[1, 0].plot(LidTightenerR_data,label=user+str(num+1))
            axs[1, 0].set_title('LidTightenerR')
            axs[1, 1].plot(UpperLipRaiserR_data,label=user+str(num+1))
            axs[1, 1].set_title('UpperLipRaiserR')
            for i in range(2):
                for j in range(2):
                    axs[i, j].legend()
    plt.tight_layout()
    plt.savefig('expression_data_drawed.png')
    plt.show()


def Head_data_drawer(studytype, user_names, dates, size_num, pin_num, repeat_num, curve_num_per_fig=6, rotdir=os.path.join(os.getcwd(),'data')):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    plt.rcParams['axes.prop_cycle'] = cycler(color=plt.cm.tab20.colors)
    plot_row=1; plot_column=3
    fig_num=0
    user_plotted=[]
    fig, axs = plt.subplots(plot_row, plot_column, figsize=(20, 10))
    for user in user_names:
        for date in dates:
            for num in range(repeat_num):
                if fig_num == 0:
                    fig, axs = plt.subplots(plot_row, plot_column, figsize=(20, 7))
                fig_num+=1
                user_plotted.append(str(user) + str(num+1))
                data=pd.read_csv(f"data{date}/P{user}/Head_data_{studytype}-{user}-{date}-" "data/Head_data_"+user+'-'+date+'-'+str(num+1)+'.csv')
                Vector3X_data=data['H-Vector3X']
                Vector3Y_data=data['H-Vector3Y']
                Vector3Z_data=data['H-Vector3Z']
                axs[0].plot(Vector3X_data, label='Head: '+user+str(num+1)+'_'+date)
                axs[0].set_title('H-Vector3X')
                axs[1].plot(Vector3Y_data, label='Head: '+user+str(num+1)+'_'+date)
                axs[1].set_title('H-Vector3Y')
                axs[2].plot(Vector3Z_data, label='Head: '+user+str(num+1)+'_'+date)
                axs[2].set_title('H-Vector3Z')
                if fig_num == curve_num_per_fig:
                    axs[0].legend()
                    if not os.path.exists("result/head_xyz"):
                        os.makedirs("result/head_xyz")
                    fig.savefig(os.path.join("result/head_xyz",re.sub(r'["\'\[\],\s]', '', "head_xyz_" + str(user_plotted) + ".png")))
                    fig_num=0
                    user_plotted=[]
                    plt.clf()
                  
def unity_angle_drawer(user_names, dates, repeat_num, curve_num_per_fig=3, source='Gaze'):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    if source=='Gaze':
        plot_row=2; plot_column=3
        fig_num=0
        user_plotted=[]
        fig, axs = plt.subplots(plot_row, plot_column, figsize=(20, 10))
        for user in user_names:
            for date in dates:
                for num in range(repeat_num):
                    if fig_num == 0:
                        fig, axs = plt.subplots(plot_row, plot_column, figsize=(20, 10))
                    fig_num+=1
                    user_plotted.append(str(user) + str(num+1))
                    data = pd.read_csv(os.path.join("data","GazeCalculate_data_" + user + "-" + date + "-" + str(num+1) + "_unity_processed.csv"))
                    df = pd.DataFrame(data)
                    axs[0,0].plot([x if abs(x) < 180 else (x - 360 if x > 180 else x + 360) for x in df['L_Yaw']], label=user+str(num+1)+'_'+date)
                    axs[0,0].set_title('L_Yaw')
                    axs[0,1].plot([x if abs(x) < 180 else (x - 360 if x > 180 else x + 360) for x in df['L_Pitch']], label=user+str(num+1)+'_'+date)
                    axs[0,1].set_title('L_Pitch')
                    axs[0,2].plot([x if abs(x) < 180 else (x - 360 if x > 180 else x + 360) for x in df['L_Roll']], label=user+str(num+1)+'_'+date)
                    axs[0,2].set_title('L_Roll')
                    axs[1,0].plot([x if abs(x) < 180 else (x - 360 if x > 180 else x + 360) for x in df['R_Yaw']], label=user+str(num+1)+'_'+date)
                    axs[1,0].set_title('R_Yaw')
                    axs[1,1].plot([x if abs(x) < 180 else (x - 360 if x > 180 else x + 360) for x in df['R_Pitch']], label=user+str(num+1)+'_'+date)
                    axs[1,1].set_title('R_Pitch')
                    axs[1,2].plot([x if abs(x) < 180 else (x - 360 if x > 180 else x + 360) for x in df['R_Roll']], label=user+str(num+1)+'_'+date)
                    axs[1,2].set_title('R_Roll')

                    if fig_num == curve_num_per_fig:
                        for i in range(plot_row):
                            for j in range(plot_column):
                                axs[i, j].legend()
                        fig.savefig(os.path.join("result/unity_angle",re.sub(r'["\'\[\],\s]', '', "unity_" + source + "_angle" + str(user_plotted) + ".png")))
                        fig_num=0
                        user_plotted=[]
                        plt.clf()
    elif source=='Head':
        plot_row=1; plot_column=3
        fig_num=0
        user_plotted=[]
        fig, axs = plt.subplots(plot_row, plot_column, figsize=(20, 10))
        for user in user_names:
            for date in dates:
                for num in range(repeat_num):
                    if fig_num == 0:
                        fig, axs = plt.subplots(plot_row, plot_column, figsize=(20, 10))
                    fig_num+=1
                    user_plotted.append(str(user) + str(num+1))
                    data = pd.read_csv(os.path.join("data","Head_data_" + user + "-" + date + "-" + str(num+1) + "_unity_processed.csv"))
                    df = pd.DataFrame(data)
                    axs[0].plot([x if abs(x) < 180 else (x - 360 if x > 180 else x + 360) for x in df['Yaw']], label=user+str(num+1)+'_'+date)
                    axs[0].set_title('Yaw')
                    axs[1].plot([x if abs(x) < 180 else (x - 360 if x > 180 else x + 360) for x in df['Pitch']], label=user+str(num+1)+'_'+date)
                    axs[1].set_title('Pitch')
                    axs[2].plot([x if abs(x) < 180 else (x - 360 if x > 180 else x + 360) for x in df['Roll']], label=user+str(num+1)+'_'+date)
                    axs[2].set_title('Roll')

                    # axs[0].plot(df['Yaw'], label=user+str(num+1)+'_'+date)
                    # axs[0].set_title('Yaw')
                    # axs[1].plot(df['Pitch'], label=user+str(num+1)+'_'+date)
                    # axs[1].set_title('Pitch')
                    # axs[2].plot(df['Roll'], label=user+str(num+1)+'_'+date)
                    # axs[2].set_title('Roll')

                    if fig_num == curve_num_per_fig:
                        for j in range(plot_column):
                            axs[j].legend()
                        fig.savefig(os.path.join("result/unity_angle",re.sub(r'["\'\[\],\s]', '', "unity_" + source + "_angle" + str(user_plotted) + ".png")))
                        fig_num=0
                        user_plotted=[]
                        plt.clf()

def difference_gaze_head_drawer(studytype_users_dates, curve_num_per_fig=3, size_num_study1= [1,2,3,4,5,6], pin_num = [1,2,3,4], authentications_per_person=6, rotdir = None): # num从1开始
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig_num=0
    user_plotted=[]
    plot_row=2; plot_column=3
    fig, axs = plt.subplots(plot_row, plot_column, figsize=(20, 10))
    for member in studytype_users_dates:
        if member.split('-')[0] == 'study1':
            authentications_per_person = 2
            size_pin_num_pair = itertools.product(range(1, len(size_num_study1)+1), range(1, len(pin_num)+1))
        
            for size in size_num_study1:
                for pin in pin_num:
                    for num in range(authentications_per_person):
                        if fig_num == 0:
                            fig, axs = plt.subplots(plot_row, plot_column, figsize=(20, 10))
                        fig_num+=1
                        user_plotted.append(f"P{member.split('-')[1]}-{member.split('-')[2]}-pin{pin}-size{size}-{num+1}_{member.split('-')[0]}")
                        Yaw_L = difference_gaze_head(member, size, pin, num+1, eye='L', angle='Yaw', rotdir=rotdir)
                        Pitch_L = difference_gaze_head(member, size, pin, num+1, eye='L', angle='Pitch', rotdir=rotdir)
                        Roll_L = difference_gaze_head(member, size, pin, num+1, eye='L', angle='Roll', rotdir=rotdir)
                        Yaw_R = difference_gaze_head(member, size, pin, num+1, eye='R', angle='Yaw', rotdir=rotdir)
                        Pitch_R = difference_gaze_head(member, size, pin, num+1, eye='R', angle='Pitch', rotdir=rotdir)
                        Roll_R = difference_gaze_head(member, size, pin, num+1, eye='R', angle='Roll', rotdir=rotdir)
                        axs[0,0].plot(Yaw_L, label=f"P{member.split('-')[1]}-{member.split('-')[2]}-pin{pin}-size{size}-{num+1}_{member.split('-')[0]}")
                        axs[0,0].set_title('L_Gaze_Yaw-Head_Yaw')
                        axs[0,1].plot(Pitch_L, label=f"P{member.split('-')[1]}-{member.split('-')[2]}-pin{pin}-size{size}-{num+1}_{member.split('-')[0]}")
                        axs[0,1].set_title('L_Gaze_Pitch-Head_Pitch')
                        axs[0,2].plot(Roll_L, label=f"P{member.split('-')[1]}-{member.split('-')[2]}-pin{pin}-size{size}-{num+1}_{member.split('-')[0]}")
                        axs[0,2].set_title('L_Gaze_Roll-Head_Roll')
                        axs[1,0].plot(Yaw_R, label=f"P{member.split('-')[1]}-{member.split('-')[2]}-pin{pin}-size{size}-{num+1}_{member.split('-')[0]}")
                        axs[1,0].set_title('R_Gaze_Yaw-Head_Yaw')
                        axs[1,1].plot(Pitch_R, label=f"P{member.split('-')[1]}-{member.split('-')[2]}-pin{pin}-size{size}-{num+1}_{member.split('-')[0]}")
                        axs[1,1].set_title('R_Gaze_Pitch-Head_Pitch')
                        axs[1,2].plot(Roll_R, label=f"P{member.split('-')[1]}-{member.split('-')[2]}-pin{pin}-size{size}-{num+1}_{member.split('-')[0]}")
                        axs[1,2].set_title('R_Gaze_Roll-Head_Roll')
                        if fig_num == curve_num_per_fig:
                            for i in range(plot_row):
                                for j in range(plot_column):
                                    axs[i, j].legend()
                            fig.savefig(os.path.join("result/difference_gaze_head",re.sub(r'["\'\[\],\s]', '', "difference_gaze_head" + str(user_plotted) + ".png")))
                            fig_num=0
                            user_plotted=[]
                            plt.clf()
    #                     diff_pitch_data = difference_gaze_head(member, size, pin, num+1, eye='L', angle='Pitch', rotdir=rotdir)
    # for user in user_names:
    #     for date in dates:
    #         for num in range(repeat_num):
    #             if fig_num == 0:
    #                 fig, axs = plt.subplots(plot_row, plot_column, figsize=(20, 10))
    #             fig_num+=1
    #             user_plotted.append(str(user) + str(num+1))
    #             Yaw_L = difference_gaze_head(user, date, num+1, eye='L', angle='Yaw')
    #             Pitch_L = difference_gaze_head(user, date, num+1, eye='L', angle='Pitch')
    #             Roll_L = difference_gaze_head(user, date, num+1, eye='L', angle='Roll')
    #             Yaw_R = difference_gaze_head(user, date, num+1, eye='R', angle='Yaw')
    #             Pitch_R = difference_gaze_head(user, date, num+1, eye='R', angle='Pitch')
    #             Roll_R = difference_gaze_head(user, date, num+1, eye='R', angle='Roll')
    
    #             axs[0,0].plot(Yaw_L, label=user+str(num+1)+'_'+date)
    #             axs[0,0].set_title('L_Gaze_Yaw-Head_Yaw')
    #             axs[0,1].plot(Pitch_L, label=user+str(num+1)+'_'+date)
    #             axs[0,1].set_title('L_Gaze_Pitch-Head_Pitch')
    #             axs[0,2].plot(Roll_L, label=user+str(num+1)+'_'+date)
    #             axs[0,2].set_title('L_Gaze_Roll-Head_Roll')
    #             axs[1,0].plot(Yaw_R, label=user+str(num+1)+'_'+date)
    #             axs[1,0].set_title('R_Gaze_Yaw-Head_Yaw')
    #             axs[1,1].plot(Pitch_R, label=user+str(num+1)+'_'+date)
    #             axs[1,1].set_title('R_Gaze_Pitch-Head_Pitch')
    #             axs[1,2].plot(Roll_R, label=user+str(num+1)+'_'+date)
    #             axs[1,2].set_title('R_Gaze_Roll-Head_Roll')
    #             if fig_num == curve_num_per_fig:
    #                 for i in range(plot_row):
    #                     for j in range(plot_column):
    #                         axs[i, j].legend()
    #                 fig.savefig(os.path.join("result/difference_gaze_head",re.sub(r'["\'\[\],\s]', '', "difference_gaze_head" + str(user_plotted) + ".png")))
    #                 fig_num=0
    #                 user_plotted=[]
    #                 plt.clf()
    
def difference_gaze_lr_euler_angle_drawer(user_names, dates, repeat_num, curve_num_per_fig=3):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    plot_row=1; plot_column=3
    fig_num=0
    user_plotted=[]
    fig, axs = plt.subplots(plot_row, plot_column, figsize=(20, 7))
    for user in user_names:
        for date in dates:
            for num in range(repeat_num):
                if fig_num == 0:
                    fig, axs = plt.subplots(plot_row, plot_column, figsize=(20, 7))
                fig_num+=1
                user_plotted.append(str(user) + str(num+1))
                L_R_Yaw, L_R_Pitch, L_R_Roll = difference_gaze_lr_euler_angle(user, date, num+1)
                axs[0].plot(L_R_Yaw, label='Gaze: '+user+str(num+1)+'_'+date)
                axs[0].set_title('L_Yaw-R_Yaw')
                axs[1].plot(L_R_Pitch, label='Gaze: '+user+str(num+1)+'_'+date)
                axs[1].set_title('L_Pitch-R_Pitch')
                axs[2].plot(L_R_Roll, label='Gaze: '+user+str(num+1)+'_'+date)
                axs[2].set_title('L_Roll-R_Roll')
                if fig_num == curve_num_per_fig:
                    axs[0].legend()
                    fig.savefig(os.path.join("result/difference_lr_angle",re.sub(r'["\'\[\],\s]', '', "difference_lr_angle" + str(user_plotted) + ".png")))
                    fig_num=0
                    user_plotted=[]
                    plt.clf()
     
def fourier_gaze_drawer(user, date, repeat_num, slice_l, slice_r, slice_l_1, slice_r_1, eye='L', angle='Yaw', save_flag=True):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    plot_row=2; plot_column=3
    for num in repeat_num:
        gaze_df, fft_gaze, freq_gaze = fourier_gaze(user, date, num, eye, angle)
        gaze_df_slice = gaze_df[slice_l:slice_r]
        fig, axs = plt.subplots(plot_row, plot_column, figsize=(20, 10))
        axs[0, 0].plot(gaze_df, label='Gaze: '+user+str(num+1)+'_'+date)
        axs[0, 0].set_title(eye + '_Gaze_' + angle)
        axs[0, 0].axvline(x=slice_l, color='r')
        axs[0, 0].axvline(x=slice_r, color='r')
        axs[1, 0].plot(freq_gaze, np.abs(fft_gaze), label='Gaze: '+user+str(num+1)+'_'+date)
        axs[1, 0].set_title(eye + '_Gaze_' + angle + '_FFT')
        axs[0, 1].plot(gaze_df_slice, color= 'r')
        axs[0, 1].set_title(eye + '_Gaze_' + angle + f'_slice_red[{slice_l}:{slice_r}]')
        fft_gaze = np.fft.fft(gaze_df_slice)
        freq_gaze=np.fft.fftfreq(len(gaze_df_slice),0.02)
        axs[1, 1].plot(freq_gaze, np.abs(fft_gaze), label='Gaze: '+user+str(num+1)+'_'+date)
        axs[1, 1].set_title(eye + '_Gaze_' + angle + '__slice_red_FFT')
        axs[1, 1].set_xlim(0, 10)
        axs[1, 1].set_ylim(0, 1200)
        axs[1, 1].set_xlabel('Frequency(Hz)')
        axs[1, 1].set_ylabel('Amplitude')

        execute_block = True # skip specific line drawing: True for skipping and False for drawing all line  
        if execute_block:
            gaze_df_slice = gaze_df[slice_l_1:slice_r_1]
            fft_gaze = np.fft.fft(gaze_df_slice)
            freq_gaze=np.fft.fftfreq(len(gaze_df_slice),0.02)
            axs[0, 0].axvline(x=slice_l_1, color='g')
            axs[0, 0].axvline(x=slice_r_1, color='g')
            axs[0, 2].plot(gaze_df_slice, color= 'g')
            axs[0, 2].set_title(eye + '_Gaze_' + angle + f'_slice_green[{slice_l_1}:{slice_r_1}]')
            axs[1, 2].plot(freq_gaze, np.abs(fft_gaze), label='Gaze: '+user+str(num+1)+'_'+date)
            axs[1, 2].set_title(eye + '_Gaze_' + angle + '__slice_red_FFT')
            axs[1, 2].set_xlim(0, 10)
            axs[1, 2].set_ylim(0, 500)
            axs[1, 2].set_xlabel('Frequency(Hz)')
            axs[1, 2].set_ylabel('Amplitude')
            axs[0, 0].legend()
            axs[1, 0].legend()
        if save_flag:
            fig.savefig(os.path.join("result/unity_processed_picture",re.sub(r'["\'\[\],\s]', '',"FFT_Gaze_angle" + str(user) + str(num+1) + ".png")))
        plt.clf()

def plot_mean_difference_gaze_head(studytype_users_dates, size_num_study1=[1,2,3,4,5,6], pin_num=[1,2,3,4], authentications_per_person=6, rotdir=None):
    # 假设 difference_gaze_head 函数已经定义，并且可以计算出gaze和head的差值
    # 首先，收集所有数据
    all_data = {}
    for size in size_num_study1:
        for pin in pin_num:
            all_data[(size, pin)] = []

    # 收集数据
    for member in studytype_users_dates:
        if member.split('-')[0] == 'study1':
            for size in size_num_study1:
                for pin in pin_num:
                    # 在这里，我们只关心gaze和head的差值，所以取一个差值作为例子
                    # 您可以根据需要修改这部分来收集您感兴趣的数据
                    difference = difference_gaze_head(member, size, pin, 1, 'L', 'Yaw', rotdir)
                    all_data[(size, pin)].append(difference)
    
    # 计算平均值
    max_length = max(len(v) for values in all_data.values() for v in values)

    # Pad the lists with np.nan to ensure they are all the same length
    for key in all_data:
        all_data[key] = [v + [np.nan]*(max_length - len(v)) for v in all_data[key]]

    # Calculate the mean, ignoring np.nan values
    mean_values = {key: np.nanmean(np.array(values), axis=0) for key, values in all_data.items()}

    # ... [rest of your code] ...
    
    # 创建条形图
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.2  # 条形的宽度
    for i, size in enumerate(size_num_study1):
        for j, pin in enumerate(pin_num):
            mean_value = mean_values[(size, pin)]
            # 为每个size和pin画一个条形
            ax.bar(i + j*width, mean_value, width, label=f'Pin {pin}')
    
    # 设置图表标题和坐标轴标签
    ax.set_title('Average Gaze-Head Difference by Size and Pin')
    ax.set_xlabel('Size')
    ax.set_ylabel('Average Difference')
    
    # 设置x轴刻度标签
    ax.set_xticks([i + width*(len(pin_num)-1)/2 for i in range(len(size_num_study1))])
    ax.set_xticklabels(size_num_study1)
    
    # 添加图例
    ax.legend(title='Pin')
    
    # 显示图表
    plt.tight_layout()
    plt.show()

# unity_angle_drawer(['mlwk','mljq','mhyr','mhyl','mgx'],['1217'],7,7)
def mean_difference_gaze_head_drawer(studytype_users_dates, size_num_study1, pin_num, authentications_per_person, rotdir):
    # 假设 difference_gaze_head 函数已经定义，并且可以计算出gaze和head的差值
    # 首先，初始化一个字典来储存所有数据
    data = {(size, pin): [] for size in size_num_study1 for pin in pin_num}

    # 收集数据
    for member in studytype_users_dates:
        if member.split('-')[0] == 'study1':
            for size in size_num_study1:
                for pin in pin_num:
                    difference = [abs(num) for num in difference_gaze_head(member, size, pin, 1, 'L', 'Roll', rotdir)]
                    data[(size, pin)].append(difference)

    # # Calculate the mean, ignoring np.nan values
    means = {key: np.nanmean([num for sublist in values for num in sublist], axis=0) for key, values in data.items()}

    # 确定每个pin的颜色
    colors = plt.cm.viridis(np.linspace(0, 1, len(pin_num)))

    # 创建每个size的条形图
    n_sizes = len(size_num_study1)
    fig, axs = plt.subplots(1, n_sizes, figsize=(n_sizes * 5, 6), squeeze=False)

    for i, size in enumerate(size_num_study1):
        for j, pin in enumerate(pin_num):
            mean_value = means.get((size, pin), 0)  # 如果没有值，则默认为0
            # print(mean_value)
            axs[0, i].bar(pin, mean_value, color=colors[j], label=f'Pin {pin}')

        axs[0, i].set_title(f'Size {size}')
        axs[0, i].set_xlabel('Pin Number')
        axs[0, i].set_ylabel('Average Difference')
        axs[0, i].legend()

    plt.tight_layout()
    plt.show()

def var_difference_gaze_head_drawer(studytype_users_dates, size_num_study1, pin_num, authentications_per_person, rotdir):
    # 假设 difference_gaze_head 函数已经定义，并且可以计算出gaze和head的差值
    # 首先，初始化一个字典来储存所有数据
    data_Yaw_L = {(size, pin): [] for size in size_num_study1 for pin in pin_num}
    data_Pitch_L = {(size, pin): [] for size in size_num_study1 for pin in pin_num}
    data_Roll_L = {(size, pin): [] for size in size_num_study1 for pin in pin_num}

    # 收集数据
    for member in studytype_users_dates:
        if member.split('-')[0] == 'study1':
            for size in size_num_study1:
                for pin in pin_num:
                    difference_Yaw_L = difference_gaze_head(member, size, pin, 1, 'L', 'Yaw', rotdir)
                    difference_Pitch_L = difference_gaze_head(member, size, pin, 1, 'L', 'Pitch', rotdir)
                    difference_Roll_L = difference_gaze_head(member, size, pin, 1, 'L', 'Roll', rotdir)
                    data_Yaw_L[(size, pin)].append(difference_Yaw_L)
                    data_Pitch_L[(size, pin)].append(difference_Pitch_L)
                    data_Roll_L[(size, pin)].append(difference_Roll_L)

    # # Calculate the var, ignoring np.nan values
    var_Yaw_L = {key: np.nanvar([num for sublist in values for num in sublist], axis=0) for key, values in data_Yaw_L.items()}
    var_Pitch_L = {key: np.nanvar([num for sublist in values for num in sublist], axis=0) for key, values in data_Pitch_L.items()}
    var_Roll_L = {key: np.nanvar([num for sublist in values for num in sublist], axis=0) for key, values in data_Roll_L.items()}

    # 确定每个pin的颜色
    colors = plt.cm.viridis(np.linspace(0, 1, len(pin_num)))

    # 创建每个size的条形图
    n_sizes = len(size_num_study1)
    fig1, axs1 = plt.subplots(1, n_sizes, figsize=(n_sizes * 5, 6), squeeze=False)
    fig2, axs2 = plt.subplots(1, n_sizes, figsize=(n_sizes * 5, 6), squeeze=False)
    fig3, axs3 = plt.subplots(1, n_sizes, figsize=(n_sizes * 5, 6), squeeze=False)

    for i, size in enumerate(size_num_study1):
        for j, pin in enumerate(pin_num):
            mean_value = var_Yaw_L.get((size, pin), 0)  # 如果没有值，则默认为0
            # print(mean_value)
            axs1[0, i].bar(pin, mean_value, color=colors[j], label=f'Pin {pin}')

        axs1[0, i].set_title(f'Size {size}')
        axs1[0, i].set_xlabel('Pin Number')
        axs1[0, i].set_ylabel('Var of Difference-Yaw_L')
        axs1[0, i].legend()

    for i, size in enumerate(size_num_study1):
        for j, pin in enumerate(pin_num):
            mean_value = var_Pitch_L.get((size, pin), 0)
            axs2[0, i].bar(pin, mean_value, color=colors[j], label=f'Pin {pin}')
        axs2[0, i].set_title(f'Size {size}')
        axs2[0, i].set_xlabel('Pin Number')
        axs2[0, i].set_ylabel('Var of Difference-Pitch_L')
        axs2[0, i].legend()

    for i, size in enumerate(size_num_study1):
        for j, pin in enumerate(pin_num):
            mean_value = var_Roll_L.get((size, pin), 0)
            axs3[0, i].bar(pin, mean_value, color=colors[j], label=f'Pin {pin}')
        axs3[0, i].set_title(f'Size {size}')
        axs3[0, i].set_xlabel('Pin Number')
        axs3[0, i].set_ylabel('Var of Difference-Roll_L')
        axs3[0, i].legend()
    plt.tight_layout()

    if not os.path.exists("result/var_difference_gaze_head"):
        os.makedirs("result/var_difference_gaze_head")
    fig1.savefig(os.path.join("result/var_difference_gaze_head",re.sub(r'["\'\[\],\s]', '', "var_difference_gaze_head_Yaw_L.png")))
    fig2.savefig(os.path.join("result/var_difference_gaze_head",re.sub(r'["\'\[\],\s]', '', "var_difference_gaze_head_Pitch_L.png")))
    fig3.savefig(os.path.join("result/var_difference_gaze_head",re.sub(r'["\'\[\],\s]', '', "var_difference_gaze_head_Roll_L.png")))

def difference_gaze_head_size_pin_drawer(studytype_users_dates, size_num_study1, pin_num, authentications_per_person, rotdir):
    # 初始化字典来储存数据
    data = {angle: {(size, pin): [] for size in size_num_study1 for pin in pin_num} 
            for angle in ['Yaw_L', 'Pitch_L', 'Roll_L']}

    # 收集数据
    print(studytype_users_dates)
    for member in studytype_users_dates:
        print(member.split('-')[0])
        if member.split('-')[0] == 'study1':
            for size in size_num_study1:
                for pin in pin_num:
                    for angle in ['Yaw_L', 'Pitch_L', 'Roll_L']:
                        for num in range(authentications_per_person):
                            differences = difference_gaze_head(member, size, pin, num+1, 'L', angle.split('_')[0], rotdir)
                            data[angle][(size, pin)].extend([abs(x) for x in differences])

    # 计算统计指标
    stats_functions = {
        'mean': np.nanmean,
        'max': np.nanmax,
        'min': np.nanmin,
        'var': np.nanvar,
        'median': np.nanmedian,
        'rms': lambda x: np.sqrt(np.nanmean(np.square(x)))  # Root Mean Square
    }
    # print(data)
    # Applying statistical functions to the data
    stats = {stat: {angle: {key: func(values) for key, values in data_angle.items()} 
                    for angle, data_angle in data.items()} 
             for stat, func in stats_functions.items()}
    # print(stats)

    # 绘制条形图
    def draw_charts(stat_data, stat_name, size_num_study1, pin_num, folder_name):
        # 确定每个pin的颜色
        colors = plt.cm.viridis(np.linspace(0, 1, len(pin_num)))
        n_sizes = len(size_num_study1)
        
        # 创建文件夹
        if not os.path.exists(f"result/{folder_name}"):
            os.makedirs(f"result/{folder_name}")
        
        # 绘制每个角度的图表
        for angle in stat_data:
            fig, axs = plt.subplots(1, n_sizes, figsize=(n_sizes * 5, 6), squeeze=False)
            for i, size in enumerate(size_num_study1):
                for j, pin in enumerate(pin_num):
                    value = stat_data[angle].get((size, pin), 0)
                    axs[0, i].bar(pin, value, color=colors[j], label=f'Pin {pin}')
                axs[0, i].set_title(f'Size {size}')
                axs[0, i].set_xlabel('Pin Number')
                axs[0, i].set_ylabel(f'{stat_name.capitalize()} of Difference-{angle}')
                axs[0, i].legend()
            plt.tight_layout()
            fig.savefig(os.path.join(f"result/{folder_name}", re.sub(r'["\'\[\],\s]', '', f"{stat_name}_difference_gaze_head_{angle}.png")))
            plt.close(fig)  # Close the figure to avoid displaying it

    # 绘制和保存所有统计图表
    for stat_name in stats_functions:
        draw_charts(stats[stat_name], stat_name, size_num_study1, pin_num, f"{stat_name}_difference_gaze_head")




def head_size_pin_drawer(studytype_users_dates, size_num_study1, pin_num, authentications_per_person, rotdir):
    # 初始化字典来储存数据
    data = {position_angle: {(size, pin): [] for size in size_num_study1 for pin in pin_num} 
            for position_angle in ['H-Vector3X', 'H-Vector3Y', 'H-Vector3Z', 'Yaw', 'Pitch', 'Roll']}

    # 收集数据
    for member in studytype_users_dates:
        if member.split('-')[0] == 'study1':
            for size in size_num_study1:
                for pin in pin_num:
                    for num in range(authentications_per_person):
                        for position_angle in ['H-Vector3X', 'H-Vector3Y', 'H-Vector3Z', 'Yaw', 'Pitch', 'Roll']:
                            if position_angle in ['H-Vector3X', 'H-Vector3Y', 'H-Vector3Z']:
                                position_angle_data = pd.read_csv(os.path.join(rotdir, f"data{member.split('-')[2]}/P{member.split('-')[1]}/Head_data_{member.split('-')[0]}-{member.split('-')[1]}-{member.split('-')[2]}-{str(size)}-{str(pin)}-{str(num+1)}.csv"))[position_angle]
                                data[position_angle][(size, pin)].extend(position_angle_data)
                            elif position_angle in ['Yaw', 'Pitch', 'Roll']:
                                position_angle_data = pd.read_csv(os.path.join(rotdir, f"data{member.split('-')[2]}/P{member.split('-')[1]}/Head_data_{member.split('-')[0]}-{member.split('-')[1]}-{member.split('-')[2]}-{str(size)}-{str(pin)}-{str(num+1)}_unity_processed.csv"))[position_angle]
                                data[position_angle][(size, pin)].extend(position_angle_data)
    for angle, size_pin_dict in data.items():
        for size_pin, values in size_pin_dict.items():
            if not values:  # This checks if the list is empty
                print(f"Empty found at angle: {angle}, size-pin: {size_pin}")
    # 计算统计指标
    stats_functions = {
        'mean': np.nanmean,
        'max': np.nanmax,
        'min': np.nanmin,
        'var': np.nanvar,
        'median': np.nanmedian,
        'rms': lambda x: np.sqrt(np.nanmean(np.square(x)))  # Root Mean Square
    }
    
    # Applying statistical functions to the data
    stats = {stat: {angle: {key: func(values) for key, values in data_angle.items()} 
                    for angle, data_angle in data.items()}
             for stat, func in stats_functions.items()}

    # 绘制条形图
    def draw_charts(stat_data, stat_name, size_num_study1, pin_num, folder_name):
        # 确定每个pin的颜色
        colors = plt.cm.viridis(np.linspace(0, 1, len(pin_num)))
        n_sizes = len(size_num_study1)
        
        # 创建文件夹
        if not os.path.exists(f"result/{folder_name}"):
            os.makedirs(f"result/{folder_name}")
        
        # 绘制每个角度的图表
        for angle in stat_data:
            fig, axs = plt.subplots(1, n_sizes, figsize=(n_sizes * 5, 6), squeeze=False)
            for i, size in enumerate(size_num_study1):
                for j, pin in enumerate(pin_num):
                    value = stat_data[angle].get((size, pin), 0)
                    axs[0, i].bar(pin, value, color=colors[j], label=f'Pin {pin}')
                axs[0, i].set_title(f'Size {size}')
                axs[0, i].set_xlabel('Pin Number')
                axs[0, i].set_ylabel(f'{stat_name.capitalize()} of Difference-{angle}')
                axs[0, i].legend()
            plt.tight_layout()
            fig.savefig(os.path.join(f"result/{folder_name}", re.sub(r'["\'\[\],\s]', '', f"{stat_name}_head_{angle}.png")))
            plt.close(fig)  # Close the figure to avoid displaying it

    # 绘制和保存所有统计图表
    for stat_name in stats_functions:
        draw_charts(stats[stat_name], stat_name, size_num_study1, pin_num, f"{stat_name}_head")
        
 
def head_statistic_data_size_drawer(studytype_users_dates, size_num_study1, pin_num, authentications_per_person, rotdir):
    # Initialize dictionary to store raw data
    raw_data = {position_angle: {(user, size, pin): []
                                 for user in studytype_users_dates for size in size_num_study1 for pin in pin_num}
                for position_angle in ['H-Vector3X', 'H-Vector3Y', 'H-Vector3Z', 'Yaw', 'Pitch', 'Roll']}

    # Collect data
    for member in studytype_users_dates:
        if member.split('-')[0] == 'study1':
            for size in size_num_study1:
                for pin in pin_num:
                    for num in range(authentications_per_person):
                        for position_angle in ['H-Vector3X', 'H-Vector3Y', 'H-Vector3Z', 'Yaw', 'Pitch', 'Roll']:
                            if position_angle in ['H-Vector3X', 'H-Vector3Y', 'H-Vector3Z']:
                                position_angle_raw_data = pd.read_csv(os.path.join(rotdir, f"data{member.split('-')[2]}/P{member.split('-')[1]}/Head_data_{member.split('-')[0]}-{member.split('-')[1]}-{member.split('-')[2]}-{str(size)}-{str(pin)}-{str(num+1)}.csv"))[position_angle]
                                position_angle_data = [x if abs(x) < 180 else (x - 360 if x > 180 else x + 360) for x in position_angle_raw_data]
                                raw_data[position_angle][(member, size, pin)].extend(position_angle_data)
                            elif position_angle in ['Yaw', 'Pitch', 'Roll']:
                                position_angle_raw_data = pd.read_csv(os.path.join(rotdir, f"data{member.split('-')[2]}/P{member.split('-')[1]}/Head_data_{member.split('-')[0]}-{member.split('-')[1]}-{member.split('-')[2]}-{str(size)}-{str(pin)}-{str(num+1)}_unity_processed.csv"))[position_angle]
                                position_angle_data = [x if abs(x) < 180 else (x - 360 if x > 180 else x + 360) for x in position_angle_raw_data]
                                raw_data[position_angle][(member, size, pin)].extend(position_angle_data)

    # Calculate statistics for each (user, size, pin)
    stats_data = {angle: {(user, size, pin): {stat: stats_functions[stat](raw_data[angle][(user, size, pin)])
                                              for stat in stats_functions}
                          for user, size, pin in raw_data[angle]}
                  for angle in raw_data}

    # Aggregate statistics: Average over all users and pins for each size
    agg_stats = {angle: {size: {stat: np.nanmean([stats_data[angle][(user, size, pin)][stat]
                                                  for user in studytype_users_dates for pin in pin_num])
                                for stat in stats_functions}
                         for size in size_num_study1}
                 for angle in stats_data}

    # Draw Charts
    def draw_charts(agg_stat_data, stat_name, size_num_study1, folder_name):
        if not os.path.exists(f"result/{folder_name}"):
            os.makedirs(f"result/{folder_name}")

        for angle in agg_stat_data:
            fig, ax = plt.subplots(figsize=(10, 6))

            x_values = list(range(len(size_num_study1)))
            y_values = [agg_stat_data[angle][size][stat_name] for size in size_num_study1]

            ax.bar(x_values, y_values, tick_label=size_num_study1)
            ax.set_title(f'{stat_name.capitalize()} of {angle}')
            ax.set_xlabel('Size')
            ax.set_ylabel(f'{stat_name.capitalize()} Value')

            plt.tight_layout()
            fig.savefig(os.path.join(f"result/{folder_name}", re.sub(r'["\'\[\],\s]', '', f"{stat_name}_{angle}.png")))
            plt.close(fig)

    for stat_name in stats_functions:
        draw_charts(agg_stats, stat_name, size_num_study1, f"{stat_name}_head_size")

# Define the statistical functions outside the function for clarity and modularity
stats_functions = {
    'mean': np.nanmean,
    'max': np.nanmax,
    'min': np.nanmin,
    'var': np.nanvar,
    'median': np.nanmedian,
    'rms': lambda x: np.sqrt(np.nanmean(np.square(x)))  # Root Mean Square
}






                        




def replace_outlier_and_smooth_data(data):
    data = replace_local_outliers(data)
    data = smooth_data(data, window_parameter= 5)
    return data








class Drawer: # 画json里的数据的图
    def __init__(self, filepath, size_list, pin_list, rotdir = os.path.join(os.getcwd(), 'data'), default_authentications_per_person=6):
        self.filepath = filepath
        self.size_list = size_list
        self.pin_list = pin_list
        self.rotdir = rotdir
        self.studytype_users_dates = self.read_data_name_from_json()
        self.default_authentications_per_person = default_authentications_per_person

    def read_data_name_from_json(self):
        with open(self.filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)
        data_list = [f"{item['studytype']}-{item['names']}-{item['date']}" for item in data['data']]
        return data_list
    


    def calculate_times(self):
        times = defaultdict(lambda: defaultdict(list))
        times_for_size_user_pin = defaultdict(lambda: defaultdict(list))
        times_for_size = defaultdict(list)
        # 创建一个空的DataFrame
        user_size_pin_time_df = pd.DataFrame(columns=['User', 'Size', 'Pin', 'Time'])
                        
        for member in self.studytype_users_dates:
            studytype, user, date = member.split('-')
            studytype='study1'
            for size in self.size_list:
                for pin in self.pin_list:
                    prefix = f"Head_data_{studytype}-{user}-{date}-{size}-{pin}-"
                    for file in os.listdir(os.path.join(self.rotdir , f"VRAuth2/P{user}")):
                        if file.startswith(prefix) and file.endswith(".csv"): # 寻找某一date-user-size-pin下的所有num的数据
                            file_path = os.path.join(self.rotdir, f"VRAuth2/P{user}", file)
                            data = pd.read_csv(file_path)
                            time = len(data) * 0.02
                            print(f"studytype: {studytype}, user: {user}, date: {date}, size: {size}, pin: {pin}, time: {time}")
                            if time <= 10 and time>=1:  # Exclude unreasonable times
                            # if True:
                                times[size][pin].append(time)
                                times_for_size[size].append(time)
                                user_size_pin_time_df = user_size_pin_time_df._append({'User': user, 'Size': size, 'Pin': pin, 'Time': time}, ignore_index=True)
                            else:
                                warnings.warn(f"Unreasonable time found: {time} seconds for {file_path}")

        for member in self.studytype_users_dates:
            studytype, user, date = member.split('-')
            for size in self.size_list:
                for pin in self.pin_list:
                    # 创建一个空列表来存储数据
                    data = []

                    # 填充数据列表
                    for size, times_ in times_for_size.items():
                        for time in times_:
                            data.append({'Size': size, 'Time': time})

                    # 创建DataFrame
                    df = pd.DataFrame(data)


        # 使用groupby计算每个Size的中位数
        median_times_df = df.groupby('Size')['Time'].median()
        mean_times_df = df.groupby('Size')['Time'].mean()
        print(f"median_times_df: {median_times_df}")
        print(f"average_times_df: {mean_times_df}")
        for member in self.studytype_users_dates:
            studytype, user, date = member.split('-')
            for size in self.size_list:
                for pin in self.pin_list:
                    sig = 0
                    for row in user_size_pin_time_df.itertuples():
                        if row[1] == user and row[2] == size and row[3] == pin:
                            sig = 1
                            break
                    if sig == 0:
                        user_size_pin_time_df = user_size_pin_time_df._append({'User': user, 'Size': size, 'Pin': pin, 'Time': median_times_df[size]}, ignore_index=True)
        print(user_size_pin_time_df)

        # Calculate average times for each (size, pin)
        average_times = defaultdict(dict)
        for size, pins in times.items():
            for pin, time_list in pins.items():
                if time_list:  # Ensure we don't divide by zero
                    average_times[size][pin] = sum(time_list) / len(time_list)
        print(f"average_times, {average_times}")
        # 执行RMANOVA
        print(user_size_pin_time_df.shape)
        user_size_pin_time_df.to_excel("user_size_pin_time_study1_sitting.xlsx", index=False)
        from statsmodels.stats.anova import AnovaRM
        aovrm = AnovaRM(user_size_pin_time_df, 'Time', 'User', within=['Size', 'Pin'])
        res = aovrm.fit()

        # 打印结果
        print(res.summary())

        return average_times, times_for_size, times_for_size_user_pin

    def plot_time_per_size(self, average_times, times_for_size):
        size_with_name = ['Small', 'Medium Small', 'Medium Large', 'Large']
        sizes = list(times_for_size.keys())
        print(f"sizes: {sizes}")
        average_times = [np.mean(times_for_size[size]) for size in self.size_list]
        se_times = [np.std(times_for_size[size]) / np.sqrt(len(times_for_size[size])) for size in self.size_list]
        sd_times = [np.std(times_for_size[size]) for size in self.size_list]
        print(f"average_times: {average_times}")
        print(f"se_times: {se_times}")
        print(f"sd_times: {sd_times}")
        # for size in self.size_list:
        #     pin_times = [average_times[size][pin] for pin in self.pin_list if pin in average_times[size].keys()]
        #     size_avg_time = sum(pin_times) / len(pin_times) if pin_times else 0  # Ensure not dividing by zero
        #     size_avg_times.append(size_avg_time)

        # Plot the bar chart and save it
        colors = plt.cm.Blues(np.linspace(0.15, 0.85, len(self.size_list)+2))
        width = 0.5
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(self.size_list, average_times, width=width, yerr=sd_times, capsize=20, color=colors, edgecolor='grey')
        # plt.bar(self.size_list, size_avg_times, color='skyblue')
        # plt.xlabel('Size')
        ax.set_ylabel('Time (s)')
        # plt.title(self.studytype_users_dates[0].split('-')[0] + 'Average Time for Different Sizes')
        ax.set_xticks(self.size_list)
        ax.set_xticklabels(size_with_name, fontsize=19)  # Ensure every size is marked
        # Remove right and top spines/borders
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.grid(True, linestyle='--', alpha=0.7, color='grey')
        fig.savefig(os.path.join(self._create_result_folder("study1"),
                                 self.studytype_users_dates[0].split('-')[0] + "average_time_for_different_sizes.png"))
        plt.close()

    def plot_time_per_size_group_by_pin(self, average_times):
        data = {}
        posthoc_results = {}
        # for pin in self.pin_list:
        #     posthoc = pg.pairwise_tests(data=df_subset, dv='Score', within='S', subject='P', padjust='bonferroni')
        #     posthoc_results[question] = posthoc
        #     for size in self.size_list:
        #         data.setdefault(pin, []).append(average_times.get(size, {}).get(pin, 0))
        
        
        # Dynamic adjustments
        num_pins = len(self.pin_list)
        num_sizes = len(self.size_list)
        total_bar_width = 0.8  # Total width for all bars in one group
        bar_width = total_bar_width / num_sizes  # Individual bar width
        pin_indices = np.arange(num_pins)  # Pin x-axis positions

        # Dynamic figure size
        fig_width = num_pins * num_sizes  # Adjust the figure width as needed
        plt.figure(figsize=(fig_width, 0.15*fig_width))  # Set a dynamic figure size
        colors = plt.cm.Blues(np.linspace(0.15, 0.85, num_sizes + 2))

        # Plotting each size's bar for each pin
        for i, size in enumerate(self.size_list):
            size -=2
            pin_times = [data[j][size] for j in self.pin_list]
            bars = plt.bar(pin_indices + i * bar_width, pin_times, width=bar_width, label=f'Size {size+1}', color = colors[i], edgecolor='grey')

            # Annotating each bar with the size label
            # for bar in bars:
            #     yval = bar.get_height()
            #     plt.text(bar.get_x() + bar.get_width() / 2, yval, f'Size {size+1}', ha='center', va='bottom')

        # Setting chart labels and title
        # plt.xlabel('Pin')
        plt.grid(axis='y', linestyle='--', alpha=0.7, color='gray')
        plt.legend(fontsize = 16)
        plt.ylabel('Average Input Time (s)', fontsize = 18)
        # plt.title(self.studytype_users_dates[0].split('-')[0] + ' Average Time for Different Pins and Sizes')
        plt.xticks(pin_indices + total_bar_width / 2 - bar_width / 2, self.pin_list, fontsize = 18)  # Center the x-ticks
        plt.xlabel('Pin', fontsize = 18)
        plt.savefig(os.path.join(self._create_result_folder("average_time_for_different_sizes_and_pins"),
                                self.studytype_users_dates[0].split('-')[0] + "average_time_for_different_pins_and_sizes_group_by_pin.png"))
        plt.close()

    def plot_time_per_pin_group_by_size(self, average_times):
        data = {}
        for size in self.size_list:
            for pin in self.pin_list:
                data.setdefault(size, []).append(average_times.get(size, {}).get(pin, 0))
         # Dynamic adjustments
        num_sizes = len(self.size_list)
        num_pins = len(self.pin_list)
        total_bar_width = 0.8  # Total width for all bars in one group
        bar_width = total_bar_width / num_pins  # Individual bar width
        size_indices = np.arange(num_sizes)  # size x-axis positions

        # Dynamic figure size
        fig_width = max(10, num_sizes * num_pins)  # Adjust the figure width as needed
        plt.figure(figsize=(fig_width, 6))  # Set a dynamic figure size

        # Plotting each pin's bar
        for i, pin in enumerate(self.pin_list):
            pin_times = [data[size][i] for size in self.size_list]
            bars = plt.bar(size_indices + i * bar_width, pin_times, width=bar_width, label=f'Pin {pin}')
            
            # Annotating each bar with the pin number
            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2, yval, f'Pin {pin}', ha='center', va='bottom')


        # Setting chart labels and title
        plt.xlabel('Size')
        plt.ylabel('Average Time (s)')
        # plt.title(self.studytype_users_dates[0].split('-')[0] + ' Average Time for Different Sizes and Pins')
        plt.xticks(size_indices + total_bar_width / 2 - bar_width / 2, self.size_list)  # Center the x-ticks

        plt.savefig(os.path.join(self._create_result_folder("study1"),
                                 self.studytype_users_dates[0].split('-')[0] + "input_time_by_size.png"))
        plt.close()

    def plot_time_per_size_boxplot(self, times_for_size):
        # 准备数据
        sizes = list(times_for_size.keys())
        times = [times_for_size[size] for size in sizes]

        # 创建一个图形对象和坐标轴
        fig, ax = plt.subplots(figsize=(8, 6))

        # 设置箱线图的箱体颜色渐变
        colors = plt.cm.Blues(np.linspace(0.15, 0.85, len(sizes) + 2))
        median_props = dict(linestyle='-', color='black')
        
        flier_props = dict(marker='d', markerfacecolor='grey', markersize=6, linestyle='none')
        # 绘制箱线图，使用patch_artist参数为箱体设置颜色
        box = ax.boxplot(times, labels=['1', '2', '3', '4'], patch_artist=True, medianprops=median_props, flierprops=flier_props)

        # 自定义每个箱体的颜色
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
        # 添加纵坐标major刻度的灰色虚线
        ax.yaxis.grid(True, linestyle='--', alpha=0.7, color='grey')

        # 设置坐标轴标签和标题
        ax.set_xlabel("Size", fontsize=14)
        ax.set_ylabel("Input Time(s)", fontsize=14)
        
        fig.savefig(os.path.join(self._create_result_folder("study1"),
                                 "input_time_boxplot.png"))
        plt.close()

        # Create a single figure with four subplots arranged horizontally
        fig, axs = plt.subplots(1, 4, figsize=(16, 4))

        for i, (size, data) in enumerate(zip(sizes, times)):
            # Calculate the parameters for the fitted normal distribution
            mu, std = stats.norm.fit(data)

            # Create a histogram on the i-th subplot with white borders
            n, bins, patches = axs[i].hist(data, bins=20, density=True, alpha=0.6, color=plt.cm.GnBu(0.8), label='Histogram', edgecolor='white') # Density是用来归一化的

            # Plot the fitted normal distribution curve on the same subplot
            y = stats.norm.pdf(bins, mu, std)
            axs[i].plot(bins, y, 'r-', linewidth=1, label='Normal Fit')

            # Add legend only to the first subplot
            if i == 0:
                axs[i].legend()

            # Remove right and top spines/borders
            axs[i].spines['right'].set_visible(False)
            axs[i].spines['top'].set_visible(False)

            # Add a gray dashed line to the y-axis major ticks on each subplot
            axs[i].yaxis.grid(True, linestyle='--', alpha=0.7, color='gray')
            axs[i].set_xlabel(f'Input Time(s)\nSize: {size-1}', fontsize=14)
            axs[i].set_ylabel('Density', fontsize=14)

        # Set the title below the subplots
        # plt.suptitle("Normality Test for Different Sizes", y=0.92)

        # Adjust the layout to prevent overlapping
        plt.tight_layout()

        # Save the combined figure as a single PNG file
        result_folder = self._create_result_folder("study1")
        plt.savefig(os.path.join(result_folder, "combined_normality_tests.png"))
        plt.close()

    # Method to plot eye data
    def eye_user_size_pin_num_drawer(self, rotdir=None, eye='L', preprocess_func=None):
        # Define angles for left and right eyes
        position_angles = {
            'L': ['L_Yaw', 'L_Pitch', 'L_Roll'],
            'R': ['R_Yaw', 'R_Pitch', 'R_Roll']
        }
        # Initialize raw data structure
        raw_data = {position_angle: {(user, size, pin): []
                                    for user in self.studytype_users_dates for size in self.size_list for pin in self.pin_list}
                    for position_angle in position_angles[eye]}

        # Collect data
        for member in self.studytype_users_dates:
            user = member.split('-')[1]  # Adjust according to how user is identified in your data
            for size in self.size_list:
                for pin in self.pin_list:
                    for num in range(self.default_authentications_per_person):
                        # Create a figure with subplots for Yaw, Pitch, and Roll
                        fig, axes = plt.subplots(3, 1, figsize=(10, 15))
                        for i, position_angle in enumerate(position_angles[eye]):
                            # Define the path to the data file
                            filename = f"GazeRaw_data_{member.split('-')[0]}-{user}-{member.split('-')[2]}-{str(size)}-{str(pin)}-{str(num+1)}.csv"
                            file_path = os.path.join(rotdir, f"VRAuthStudy1Angle-{member.split('-')[2]}/P{user}/{filename}")
                            
                            # Load data
                            position_angle_raw_data = pd.read_csv(file_path)[position_angle]
                            position_angle_data = [x if abs(x) < 180 else (x - 360 if x > 180 else x + 360) for x in position_angle_raw_data]
                            raw_data[position_angle][(member, size, pin)].extend(position_angle_data)
                            
                            # Plotting each angle in its subplot
                            ax = axes[i]
                            if preprocess_func:
                                ax.plot(preprocess_func(position_angle_data))
                            else:
                                ax.plot(position_angle_data)
                            ax.set_title(f"{position_angle} for {member.split('-')[0]} User {user}, Date{member.split('-')[2]}, Size {size}, Pin {pin}, Auth number {num+1}")
                            ax.set_xlabel("Time")
                            ax.set_ylabel(position_angle)
                        
                        # Save plot
                        plot_folder = os.path.join("result/", "timeseries_plots", f"{member.split('-')[0]}", "gaze")
                        if not os.path.exists(plot_folder):
                            os.makedirs(plot_folder)
                        
                        if preprocess_func:
                            plot_filename = f"GazeRaw_{eye}_User{user}_Date{member.split('-')[2]}_Size{size}_Pin{pin}_Num{num+1} {preprocess_func.__name__}.png"
                        else:
                            plot_filename = f"GazeRaw_{eye}_User{user}_Date{member.split('-')[2]}_Size{size}_Pin{pin}_Num{num+1}.png"
                        fig.savefig(os.path.join(plot_folder, plot_filename))
                        plt.close(fig)

    # Method to plot head data
    def head_user_size_pin_num_drawer(self, rotdir=None, preprocess_func=None):
        # Initialize raw data structure for H-Vectors and Angles
        h_vectors = ['H-Vector3X', 'H-Vector3Y', 'H-Vector3Z']
        angles = ['Yaw', 'Pitch', 'Roll']
        raw_data = {position_angle: {(user, size, pin): []
                                    for user in self.studytype_users_dates for size in self.size_list for pin in self.pin_list}
                    for position_angle in h_vectors + angles}

        # Collect data and save time-series plots
        for member in self.studytype_users_dates:
            user = member.split('-')[1]  # Adjust according to how user is identified in your data
            for size in self.size_list:
                for pin in self.pin_list:
                    for num in range(self.default_authentications_per_person):
                        # Create figures for H-Vectors and Angles
                        fig_hvectors, axes_hvectors = plt.subplots(3, 1, figsize=(10, 15))
                        fig_angles, axes_angles = plt.subplots(3, 1, figsize=(10, 15))

                        # Plotting for H-Vectors and Angles
                        for i, position_angle in enumerate(h_vectors + angles):
                            # Define file path based on angle type
                            if position_angle in h_vectors:
                                filename = f"Head_data_{member.split('-')[0]}-{user}-{member.split('-')[2]}-{str(size)}-{str(pin)}-{str(num+1)}.csv"
                                file_path = os.path.join(rotdir, f"VRAuthStudy1-{member.split('-')[2]}/P{user}/{filename}")
                            if position_angle in angles:
                                filename = f"Head_data_{member.split('-')[0]}-{user}-{member.split('-')[2]}-{str(size)}-{str(pin)}-{str(num+1)}.csv"
                                file_path = os.path.join(rotdir, f"VRAuthStudy1Angle-{member.split('-')[2]}/P{user}/{filename}")

                            # Load data
                            position_angle_raw_data = pd.read_csv(file_path)[position_angle]
                            if position_angle in angles:
                                position_angle_data = [x if abs(x) < 180 else (x - 360 if x > 180 else x + 360) for x in position_angle_raw_data]
                            else:
                                position_angle_data = position_angle_raw_data
                            
                            raw_data[position_angle][(member, size, pin)].extend(position_angle_data)

                            
                            # Select the appropriate subplot and plot the data
                            if position_angle in h_vectors:
                                ax = axes_hvectors[h_vectors.index(position_angle)]
                                fig_to_save = fig_hvectors
                            else:
                                ax = axes_angles[angles.index(position_angle)]
                                fig_to_save = fig_angles

                            if preprocess_func:
                                ax.plot(preprocess_func(position_angle_data))
                            else:
                                ax.plot(position_angle_data)
                            ax.set_title(f"{position_angle} for User {user}, Size {size}, Pin {pin}, Auth number {num+1}")
                            ax.set_xlabel("Time")
                            ax.set_ylabel(position_angle)

                        # Save plots for each type
                        for fig, angle_type in zip([fig_hvectors, fig_angles], ['H-Vector', 'Angles']):
                            plot_folder = os.path.join("result/", "timeseries_plots", f"{member.split('-')[0]}", "head")
                            if not os.path.exists(plot_folder):
                                os.makedirs(plot_folder)
                            
                            if preprocess_func:
                                plot_filename = f"Head_{angle_type}_User{user}_Date{member.split('-')[2]}_Size{size}_Pin{pin}_Num{num+1} {preprocess_func.__name__}.png"
                            else:
                                plot_filename = f"Head_{angle_type}_User{user}_Date{member.split('-')[2]}_Size{size}_Pin{pin}_Num{num+1}.png"
                            fig.savefig(os.path.join(plot_folder, plot_filename))
                            plt.close(fig)

    def head_and_eye_drawer(self, rotdir=None, preprocess_func=None):
        # Define angles for left eye and head
        eye_angles = ['L_Yaw', 'L_Pitch', 'L_Roll']
        head_angles = ['Yaw', 'Pitch', 'Roll']

        # Loop to process and plot data
        for member in self.studytype_users_dates:
            user = member.split('-')[1]  # Adjust according to how user is identified in your data
            for size in self.size_list:
                for pin in self.pin_list:
                    for num in range(self.default_authentications_per_person):
                        # Create a figure with subplots for each angle comparison
                        # fig, axes = plt.subplots(3, 1, figsize=(10, 15))
                        fig, axes = plt.subplots(1, 3, figsize=(30, 5))

                        # Determine the path for the specific text file
                        text_filename = f"Saccades_{member}-{size}-{pin}-{num+1}.txt"
                        text_file_path = os.path.join(rotdir, f"VRAuthStudy1-{member.split('-')[2]}/P{user}/{text_filename}")
                        # Read and parse text data from the file
                        try:
                            with open(text_file_path, 'r') as file:
                                text_data = file.read().strip()
                                # Parse the ranges from the text data
                                ranges = [list(map(int, r.split('-'))) for r in text_data.split(';') if r]
                        except FileNotFoundError:
                            ranges = []  # No ranges to add if file is not found


                        for i, (eye_angle, head_angle) in enumerate(zip(eye_angles, head_angles)):
                            # Eye data path
                            eye_filename = f"GazeCalculate_data_{member.split('-')[0]}-{user}-{member.split('-')[2]}-{str(size)}-{str(pin)}-{str(num+1)}.csv"
                            eye_file_path = os.path.join(rotdir, f"VRAuthStudy1Angle-{member.split('-')[2]}/P{user}/{eye_filename}")

                            # Head data path
                            head_filename = f"Head_data_{member.split('-')[0]}-{user}-{member.split('-')[2]}-{str(size)}-{str(pin)}-{str(num+1)}.csv"
                            head_file_path = os.path.join(rotdir, f"VRAuthStudy1Angle-{member.split('-')[2]}/P{user}/{head_filename}")

                            # Load eye and head data
                            eye_data = pd.read_csv(eye_file_path)[eye_angle]
                            head_data = pd.read_csv(head_file_path)[head_angle]

                            # Preprocess and adjust the angles if necessary
                            eye_data_adjusted = [x if abs(x) < 180 else (x - 360 if x > 180 else x + 360) for x in eye_data]
                            head_data_adjusted = [x if abs(x) < 180 else (x - 360 if x > 180 else x + 360) for x in head_data]

                            # Plotting each angle on the same subplot
                            ax = axes[i]
                            if preprocess_func:
                                ax.plot(preprocess_func(eye_data_adjusted), label=f"{eye_angle} (Eye)")
                                ax.plot(preprocess_func(head_data_adjusted), label=f"{head_angle} (Head)")
                            else:
                                ax.plot(eye_data_adjusted, label=f"{eye_angle} (Eye)")
                                ax.plot(head_data_adjusted, label=f"{head_angle} (Head)")

                            # Add vertical lines for each range
                            for start, end in ranges:
                                ax.axvline(x=start, color='r', linestyle='--')
                                ax.axvline(x=end, color='r', linestyle='--')
                                ax.axvspan(start, end, color='grey', alpha=0.3)

                            ax.legend()
                            ax.set_title(f"Euler Angle for User {user}, Date {member.split('-')[2]}, Size {size}, Pin {pin}, Auth number {num+1}")
                            ax.set_xlabel("Time")
                            ax.set_ylabel("Angle")

                        # Define and create the plot folder
                        plot_folder = os.path.join("result/", "timeseries_plots", f"{member.split('-')[0]}", "combined")
                        if not os.path.exists(plot_folder):
                            os.makedirs(plot_folder)

                        # Define the plot filename
                        plot_filename = f"Combined_Eye_Head_User{user}_Date{member.split('-')[2]}_Size{size}_Pin{pin}_Num{num+1}_Angle_horizontal.png"
                        fig.savefig(os.path.join(plot_folder, plot_filename))
                        plt.close(fig)

    def pin_error_rate(self, rotdir=None):

        def remove_duplicates(input_str):
            seen = set()
            result = []
            
            for char in input_str:
                if char not in seen:
                    seen.add(char)
                    result.append(char)
            
            return ''.join(result)
        missing_check_dict = {'68':'678', '86':'876', '35':'345', '53':'543', '02': '012', '20': '210',
                        '06': '036', '60': '630', '17':'147', '71':'741', '28':'258', '82':'852',
                        '62': '642', '26': '246', '80': '840', '08': '048'}
        
        pin1 = "67852"
        pin2 = "6785"
        pin3 = "6784"
        pin4 = "6745"
        pin5 = "6425"
        pin6 = "04852"
        pin7 = "63012"
        pin8 = "84012"
        pin9 = "67840"
        pin10 = "3012"
        pin11 = "67412"
        pin12 = "7412"
        pin13 = "6784012"
        pin14 = "64258"
        pin15 = "730125"
        pin16 = "6785210"
        pin17 = "6421"
        pin18 = "48523"

        correct_pin_dict = {1: pin1, 2: pin2, 3: pin3, 4: pin4, 5: pin5, 6: pin6, 7: pin7, 8: pin8, 9: pin9,
                    10: pin10, 11: pin11, 12: pin12, 13: pin13, 14: pin14, 15: pin15, 16: pin16, 17: pin17, 18: pin18}
        
        # 储存每个pin在每个size下的输入结果
        input_pin_result_dict = defaultdict(lambda: defaultdict(list)) # {size: {pin: [input_pin1, input_pin2, ...]}}

        # 储存完全匹配的每个结果的准确率
        correct_pin_accuracy_dict = defaultdict(lambda: defaultdict(list)) # {size: {pin: [accuracy1, accuracy2, ...]}}

        # 储存编辑距离的每个输入结果的准确率
        edit_distance_accuracy_dict = defaultdict(lambda: defaultdict(list)) # {size: {pin: [accuracy1, accuracy2, ...]}}

        # 储存自串匹配的每个输入结果的准确率
        substring_accuracy_dict = defaultdict(lambda: defaultdict(list)) # {size: {pin: [accuracy1, accuracy2, ...]}}

        # Loop to process and plot data
        for member in self.studytype_users_dates:
            # study_type = member.split('-')[0]
            study_type = 'study1'
            date = member.split('-')[2]
            user = member.split('-')[1]  # Adjust according to how user is identified in your data
            for size in self.size_list:
                for pin in self.pin_list:
                    for num in range(self.default_authentications_per_person):
                        # Create a figure with subplots for each angle comparison
                        # fig, axes = plt.subplots(3, 1, figsize=(10, 15))
                        # fig, axes = plt.subplots(1, 3, figsize=(30, 5))

                        # Determine the path for the specific text file
                        text_filename = f"PINEntry_{study_type}-{user}-{date}-{size}-{pin}-{num+1}.txt"
                        text_file_path = os.path.join(rotdir, f"VRAuth2/P{user}/{text_filename}")
                        # Read and parse text data from the file
                        try:
                            with open(text_file_path, 'r') as file:
                                text_data = file.read().strip()
                                # print(f"text_data: {text_data}")
                                input_pin = ''.join([i for i in text_data.split('-') if i]) # get pin input string like '684012'
                                    # Parse the ranges from the text data
                                    # input_pin = [list(map(int, r.split('-'))) for r in text_data.split(';') if r]
                        except FileNotFoundError:
                            warnings.warn(f"File {text_file_path} not found")
                            # input_pin = []  # No input_pin to add if file is not found
                            continue
                        
                        for i in range(len(input_pin)-1): # 补充三点共线时中间点未识别的情况
                            for key, value in missing_check_dict.items():
                                if input_pin[i] + input_pin[i+1] == key:
                                    input_pin = input_pin[:i] + value + input_pin[i+2:]
                                    break
                        input_pin = remove_duplicates(input_pin)
                        print(f"studytype: {study_type}, date: {date}, user: {user}, size: {size}, pin: {pin}, num: {num+1}, input_pin: {input_pin}")
                        
                        input_pin_result_dict[size][pin].append(input_pin)
                        # save input_pin_resul_dict to .txt file
                        # 假设文件名为output.txt
                        # output_filename = "input_pin_resul_dict.txt"

                        # 将字典保存为JSON文件
                        # with open(output_filename, 'w') as file:
                        #     json.dump(input_pin_result_dict, file)
                        
                        
        # 计算每个size下每个pin的错误率
        for size, pin_dict in input_pin_result_dict.items():
            for pin, input_pin_list in pin_dict.items():
                for input_pin_str in input_pin_list:
                    correct_pin = correct_pin_dict[pin]
                    # 计算完全匹配的准确率
                    # correct_pin_accuracy_dict[size][pin].append( 1 if input_pin_str == correct_pin else 0 ) #accuracy
                    correct_pin_accuracy_dict[size][pin].append( 0 if input_pin_str == correct_pin else 1 ) # error rate
                    # 计算最长字串匹配的准确率
                    # substring_accuracy_dict[size][pin].append(SequenceMatcher(None, input_pin_str, correct_pin).find_longest_match(0, len(input_pin_str), 0, len(correct_pin)).size / len(correct_pin)) # acccuracy
                    substring_accuracy_dict[size][pin].append(1 - SequenceMatcher(None, input_pin_str, correct_pin).find_longest_match(0, len(input_pin_str), 0, len(correct_pin)).size / len(correct_pin)) # error rate
                    print(f"date: {date}, user: {user}, size: {size}, pin: {pin}, correct_pin: {correct_pin}, input_pin_str: {input_pin_str}, correct_accuracy: {correct_pin_accuracy_dict[size][pin]}, substring_accuracy: {substring_accuracy_dict[size][pin]}")
        

        
        # Convert your accuracy data to a suitable format for plotting
        data_for_plotting = []
        for size, pin_dict in correct_pin_accuracy_dict.items():
            for pin, accuracies_correct in pin_dict.items():
                for accuracy_correct in accuracies_correct:
                    data_for_plotting.append({'Size': size - 1, 'PIN': pin, 'Input Error Rate': accuracy_correct*100})

        # Create a DataFrame for easy plotting
        df = pd.DataFrame(data_for_plotting)

        df.to_excel(os.path.join(os.getcwd(), 'data','VRAuth1', 'error_rate_correct_match_usability_study3_standing.xlsx'), index=False)

        # 完全匹配
        # 画每个size下每个pin的正确率的箱型图，grouped by pin
        # Plotting
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='PIN', y='Input Error Rate', hue='Size', data=df)
        plt.title('Box Plot of Correct PIN Accuracy for Each PIN, Grouped by Size')
        result_folder = self._create_result_folder("study1")
        plt.savefig(os.path.join(result_folder, "correct_error_rate_size_pin_group_by_pin.png"))
        plt.close()
        
        
        # 画每个size下每个pin的正确率的箱型图，grouped by size
        plt.figure(figsize = (12, 6))
        sns.boxplot(x='Size', y='Input Error Rate', hue='PIN', data=df)
        plt.title('Box Plot of Correct PIN Accuracy for Each Size, Grouped by PIN')
        result_folder = self._create_result_folder("study1")
        plt.savefig(os.path.join(result_folder, "correct_error_rate_size_pin_group_by_size.png"))
        plt.close()
                
        # 画每个size下的正确率的bar
        plt.figure(figsize = (12, 6))
        sns.boxplot(x='Size', y='Input Error Rate', data=df)
        plt.title('Box Plot of Correct PIN Accuracy for Each Size')
        result_folder = self._create_result_folder("study1")
        plt.savefig(os.path.join(result_folder, "correct_error_rate_size_box.png"))
        plt.close()

        # 创建一个图形对象和坐标轴
        fig, ax = plt.subplots(figsize=(8, 6))

        # 计算每个Size的平均Input Error Rate及其标准误差
        sizes = df['Size'].unique()
        means = [df[df['Size'] == size]['Input Error Rate'].mean() for size in sizes]
        std_errs = [df[df['Size'] == size]['Input Error Rate'].sem() for size in sizes]
        print(f"means: {means}, std_errs: {std_errs}")

        # # 绘制条形图并添加误差线
        # bar_width = 0.5
        # bars = ax.bar(sizes, means, width=bar_width, color='skyblue', edgecolor='black', yerr=std_errs, capsize=5)

        # 设置箱线图的箱体颜色渐变
        colors = plt.cm.Blues(np.linspace(0.2, 1, len(self.size_list)+3))
        # 在每个条形上方显示数值
        # 绘制带有标准误差线的条形图
        bar_width = 0.7
        for i, (mean, std_err, color) in enumerate(zip(means, std_errs, colors)):
            ax.bar(sizes[i], mean, yerr=std_err, capsize=14, color=color, edgecolor='black', width=bar_width)

            # # 在每个条形上方显示数值
            # ax.annotate(f'{mean:.2f}',
            #             xy=(sizes[i] + bar_width/4, mean),
            #             xytext=(0, 3),  # 3点垂直偏移
            #             textcoords="offset points",
            #             ha='center', va='bottom', fontsize=16)



        
        # # colors = plt.cm.Blues(np.linspace(0.15, 0.85, len(sizes) + 2))
        # median_props = dict(linestyle='-', color='black')
        
        # flier_props = dict(marker='d', markerfacecolor='grey', markersize=6, linestyle='none')
        # # 绘制箱线图，使用patch_artist参数为箱体设置颜色
        # box = ax.boxplot([df[df['Size'] == size]['Input Error Rate'] for size in sizes], labels=['1', '2', '3', '4'], patch_artist=True, medianprops=median_props, flierprops=flier_props)

        # # 自定义每个箱体的颜色
        # for patch, color in zip(box['boxes'], colors):
        #     patch.set_facecolor(color)
        # 设置X轴刻度只显示整数
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # # 添加纵坐标major刻度的灰色虚线
        ax.yaxis.grid(True, linestyle='--', alpha=0.7, color='grey')

    


        # ax = sns.barplot(x='Size', y='Input Error Rate', data=df, palette=colors, errorbar='se', width=0.5, errcolor='grey')

        # # 设置误差帽
        # for bar in ax.patches:
        #     x = bar.get_x() + bar.get_width() / 2
        #     y = bar.get_height()
        #     error = df[df['Size'] == bar.get_x()]['Accuracy_correct'].std()  # 计算标准差
        #     plt.errorbar(x, y, yerr=error, fmt='none', capsize=10, color='black')
        # 在每个条形的顶端添加平均值标签
        # for p in ax.patches:
        #     ax.annotate(f"{p.get_height():.3f}",  # 设置标签格式
        #                 (p.get_x() + p.get_width() * 0.75, p.get_height()),  # 设置标签位置
        #                 ha = 'center', va = 'center',  # 水平居中、垂直居中
        #                 xytext = (0, 9),  # 文本偏移量
        #                 textcoords = 'offset points', fontsize = 14)
        # Remove right and top spines/borders
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Add a gray dashed line to the y-axis major ticks on each subplot
        # ax.yaxis.grid(True, linestyle='--', alpha=0.7, color='gray')
        ax.set_xlabel(f'Size\n(a) Exact Matching')
        ax.set_ylabel('Input Error Rate(%)')
        # plt.title('')
        result_folder = self._create_result_folder("study1")
        fig.tight_layout()
        fig.savefig(os.path.join(result_folder, "correct_error_rate_size_bar.png"))
        plt.close()

         # Convert your accuracy data to a suitable format for plotting
        data_for_plotting = []
        for size, pin_dict in substring_accuracy_dict.items():
            for pin, accuracies_substring in pin_dict.items():
                for accuracy_substring in accuracies_substring:
                    data_for_plotting.append({'Size': size - 1, 'PIN': pin, 'Input Error Rate': accuracy_substring*100})

        # Create a DataFrame for easy plotting
        df = pd.DataFrame(data_for_plotting)

        df.to_excel(os.path.join(os.getcwd(), 'data','VRAuth1', 'error_rate_substring_usability_study3_standing.xlsx'), index=False)

        # 最长子串匹配
        # 画每个size下每个pin的正确率的箱型图，grouped by pin
        # Plotting
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='PIN', y='Input Error Rate', hue='Size', data=df)
        plt.title('Box Plot of Substring PIN Accuracy for Each PIN, Grouped by Size')
        result_folder = self._create_result_folder("study1")
        plt.savefig(os.path.join(result_folder, "substring_error_rate_size_pin_group_by_pin.png"))
        plt.close()
        
        # 画每个size下每个pin的正确率的箱型图，grouped by size
        plt.figure(figsize = (12, 6))
        sns.boxplot(x='Size', y='Input Error Rate', hue='PIN', data=df)
        plt.title('Box Plot of Substring PIN Accuracy for Each Size, Grouped by PIN')
        result_folder = self._create_result_folder("study1")
        plt.savefig(os.path.join(result_folder, "substring_error_rate_size_pin_group_by_size.png"))
        plt.close()
                
        # 画每个size下的正确率的箱型图
        plt.figure(figsize = (12, 6))
        sns.boxplot(x='Size', y='Input Error Rate', data=df)
        plt.title('Box Plot of Substring PIN Accuracy for Each Size')
        result_folder = self._create_result_folder("study1")
        plt.savefig(os.path.join(result_folder, "substring_error_rate_size_box.png"))
        plt.close()



        # # 画每个size下平均正确率的条形图，用sns
        # plt.figure(figsize = (10, 6))
        # # 绘制条形图，为每个条形指定颜色
        
        # ax = sns.barplot(x='Size', y='Input Error Rate', data=df, palette=colors, errorbar='se', width=0.5, errcolor='grey')
        # for p in ax.patches:
        #     ax.annotate(f"{p.get_height():.3f}",  # 设置标签格式
        #                 (p.get_x() + p.get_width() * 0.75, p.get_height()),  # 设置标签位置
        #                 ha = 'center', va = 'center',  # 水平居中、垂直居中
        #                 xytext = (0, 9),  # 文本偏移量
        #                 textcoords = 'offset points', fontsize = 14)
        # # Remove right and top spines/borders
        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)

        # # Add a gray dashed line to the y-axis major ticks on each subplot
        # ax.yaxis.grid(True, linestyle='--', alpha=0.7, color='gray')
        # ax.set_xlabel(f'Size', fontsize=14)
        # ax.set_ylabel('Input Error Rate(%)\nBar Plot of Average Pattern Matching Error Rate for Each Size', fontsize=14)
        # # plt.title('') 
        # result_folder = self._create_result_folder("study1")
        # plt.tight_layout()
        # plt.savefig(os.path.join(result_folder, "substring_error_rate_size_bar.png"))
        # plt.close()

        # ---------bar plot---------
         # 创建一个图形对象和坐标轴
        fig, ax = plt.subplots(figsize=(8, 6))

        # 计算每个Size的平均Input Error Rate及其标准误差
        sizes = df['Size'].unique()
        means = [df[df['Size'] == size]['Input Error Rate'].mean() for size in sizes]
        std_errs = [df[df['Size'] == size]['Input Error Rate'].sem() for size in sizes]
        print(f"means: {means}, std_errs: {std_errs}")

        # # 绘制条形图并添加误差线
        # bar_width = 0.5
        # bars = ax.bar(sizes, means, width=bar_width, color='skyblue', edgecolor='black', yerr=std_errs, capsize=5)

        # 设置箱线图的箱体颜色渐变
        colors = plt.cm.Blues(np.linspace(0.2, 1, len(self.size_list)+3))
        # 在每个条形上方显示数值
        # 绘制带有标准误差线的条形图
        bar_width = 0.7
        for i, (mean, std_err, color) in enumerate(zip(means, std_errs, colors)):
            ax.bar(sizes[i], mean, yerr=std_err, capsize=14, color=color, edgecolor='black', width=bar_width)

            # 在每个条形上方显示数值
            # ax.annotate(f'{mean:.2f}',
            #             xy=(sizes[i] + bar_width/4, mean),
            #             xytext=(0, 3),  # 3点垂直偏移
            #             textcoords="offset points",
            #             ha='center', va='bottom', fontsize=16)
        # 设置X轴刻度只显示整数
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # # 添加纵坐标major刻度的灰色虚线
        ax.yaxis.grid(True, linestyle='--', alpha=0.7, color='grey')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Add a gray dashed line to the y-axis major ticks on each subplot
        # ax.yaxis.grid(True, linestyle='--', alpha=0.7, color='gray')
        ax.set_xlabel(f'Size\n(b) Pattern Matching')
        ax.set_ylabel('Input Error Rate(%)')
        # plt.title('')
        result_folder = self._create_result_folder("study1")
        fig.tight_layout()
        fig.savefig(os.path.join(result_folder, "substring_error_rate_size_bar.png"))
        plt.close()


        # ---------box plot---------
        # # 创建一个图形对象和坐标轴
        # fig, ax = plt.subplots(figsize=(8, 6))
        # # 设置箱线图的箱体颜色渐变
        # colors = plt.cm.Blues(np.linspace(0.2, 1, len(self.size_list)+3))
        # # colors = plt.cm.Blues(np.linspace(0.15, 0.85, len(sizes) + 2))
        # median_props = dict(linestyle='-', color='black')
        # flier_props = dict(marker='d', markerfacecolor='grey', markersize=6, linestyle='none')
        # # 绘制箱线图，使用patch_artist参数为箱体设置颜色
        # sizes = df['Size'].unique()
        # box = ax.boxplot([df[df['Size'] == size]['Input Error Rate'] for size in sizes], labels=['1', '2', '3', '4'], patch_artist=True, medianprops=median_props, flierprops=flier_props)
        # # 自定义每个箱体的颜色
        # for patch, color in zip(box['boxes'], colors):
        #     patch.set_facecolor(color)
        # # 添加纵坐标major刻度的灰色虚线
        # ax.yaxis.grid(True, linestyle='--', alpha=0.7, color='grey')
        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)
        # # Add a gray dashed line to the y-axis major ticks on each subplot
        # # ax.yaxis.grid(True, linestyle='--', alpha=0.7, color='gray')
        # ax.set_xlabel(f'Size', fontsize=14)
        # ax.set_ylabel('Input Error Rate(%)\nBar Plot of Average Exact Matching Error Rate for Each Size', fontsize=14)
        # # plt.title('')
        # result_folder = self._create_result_folder("study1")
        # fig.tight_layout()
        # fig.savefig(os.path.join(result_folder, "substring_error_rate_size_bar.png"))
        # plt.close()
        # ---------box plot---------


                        # for i, (eye_angle, head_angle) in enumerate(zip(eye_angles, head_angles)):
                        #     # Eye data path
                        #     eye_filename = f"GazeCalculate_data_{member.split('-')[0]}-{user}-{member.split('-')[2]}-{str(size)}-{str(pin)}-{str(num+1)}.csv"
                        #     eye_file_path = os.path.join(rotdir, f"VRAuthStudy1Angle-{member.split('-')[2]}/P{user}/{eye_filename}")

                        #     # Head data path
                        #     head_filename = f"Head_data_{member.split('-')[0]}-{user}-{member.split('-')[2]}-{str(size)}-{str(pin)}-{str(num+1)}.csv"
                        #     head_file_path = os.path.join(rotdir, f"VRAuthStudy1Angle-{member.split('-')[2]}/P{user}/{head_filename}")

                        #     # Load eye and head data
                        #     eye_data = pd.read_csv(eye_file_path)[eye_angle]
                        #     head_data = pd.read_csv(head_file_path)[head_angle]

                        #     # Preprocess and adjust the angles if necessary
                        #     eye_data_adjusted = [x if abs(x) < 180 else (x - 360 if x > 180 else x + 360) for x in eye_data]
                        #     head_data_adjusted = [x if abs(x) < 180 else (x - 360 if x > 180 else x + 360) for x in head_data]

                        #     # Plotting each angle on the same subplot
                        #     ax = axes[i]
                        #     if preprocess_func:
                        #         ax.plot(preprocess_func(eye_data_adjusted), label=f"{eye_angle} (Eye)")
                        #         ax.plot(preprocess_func(head_data_adjusted), label=f"{head_angle} (Head)")
                        #     else:
                        #         ax.plot(eye_data_adjusted, label=f"{eye_angle} (Eye)")
                        #         ax.plot(head_data_adjusted, label=f"{head_angle} (Head)")

                        #     # Add vertical lines for each range
                        #     for start, end in ranges:
                        #         ax.axvline(x=start, color='r', linestyle='--')
                        #         ax.axvline(x=end, color='r', linestyle='--')
                        #         ax.axvspan(start, end, color='grey', alpha=0.3)

                        #     ax.legend()
                        #     ax.set_title(f"Euler Angle for User {user}, Date {member.split('-')[2]}, Size {size}, Pin {pin}, Auth number {num+1}")
                        #     ax.set_xlabel("Time")
                        #     ax.set_ylabel("Angle")

                        # # Define and create the plot folder
                        # plot_folder = os.path.join("result/", "timeseries_plots", f"{member.split('-')[0]}", "combined")
                        # if not os.path.exists(plot_folder):
                        #     os.makedirs(plot_folder)

                        # # Define the plot filename
                        # plot_filename = f"Combined_Eye_Head_User{user}_Date{member.split('-')[2]}_Size{size}_Pin{pin}_Num{num+1}_Angle_horizontal.png"
                        # fig.savefig(os.path.join(plot_folder, plot_filename))
                        # plt.close(fig)

    def head_and_eye_quaternion_drawer(self, rotdir=None, eye_preprocess_func=None, head_preprocess_func=None, file_suffix = None):
        # Define angles for left eye and head
        eye_angles = ['L-QuaternionX', 'L-QuaternionY', 'L-QuaternionZ', 'L-QuaternionW']
        head_angles = ['H-QuaternionX', 'H-QuaternionY', 'H-QuaternionZ', 'H-QuaternionW']

        # Loop to process and plot data
        for member in self.studytype_users_dates:
            user = member.split('-')[1]  # Adjust according to how user is identified in your data
            for size in self.size_list:
                for pin in self.pin_list:
                    for num in range(self.default_authentications_per_person):
                        # Create a figure with subplots for each angle comparison
                        fig, axes = plt.subplots(4, 1, figsize=(10, 18))

                        # Determine the path for the specific text file
                        text_filename = f"Saccades_{member}-{size}-{pin}-{num+1}.txt"
                        text_file_path = os.path.join(rotdir, f"VRAuthStudy1-{member.split('-')[2]}/P{user}/{text_filename}")
                        # Read and parse text data from the file
                        try:
                            with open(text_file_path, 'r') as file:
                                text_data = file.read().strip()
                                # Parse the ranges from the text data
                                ranges = [list(map(int, r.split('-'))) for r in text_data.split(';') if r]
                        except FileNotFoundError:
                            ranges = []  # No ranges to add if file is not found


                        for i, (eye_angle, head_angle) in enumerate(zip(eye_angles, head_angles)):
                            # Eye data path
                            eye_filename = f"GazeCalculate_data_{member.split('-')[0]}-{user}-{member.split('-')[2]}-{str(size)}-{str(pin)}-{str(num+1)}.csv"
                            eye_file_path = os.path.join(rotdir, f"VRAuthStudy1-{member.split('-')[2]}/P{user}/{eye_filename}")

                            # Head data path
                            head_filename = f"Head_data_{member.split('-')[0]}-{user}-{member.split('-')[2]}-{str(size)}-{str(pin)}-{str(num+1)}.csv"
                            head_file_path = os.path.join(rotdir, f"VRAuthStudy1-{member.split('-')[2]}/P{user}/{head_filename}")

                            # Load eye and head data
                            eye_data = pd.read_csv(eye_file_path)[eye_angle]
                            head_data = pd.read_csv(head_file_path)[head_angle]

                            # Preprocess and adjust the angles if necessary
                            eye_data_adjusted = [x if abs(x) < 180 else (x - 360 if x > 180 else x + 360) for x in eye_data]
                            head_data_adjusted = [x if abs(x) < 180 else (x - 360 if x > 180 else x + 360) for x in head_data]

                            # Plotting each angle on the same subplot
                            ax = axes[i]
                            if eye_preprocess_func:
                                ax.plot(eye_preprocess_func(eye_data_adjusted), label=f"{eye_angle} (Eye)")
                            else:
                                ax.plot(eye_data_adjusted, label=f"{eye_angle} (Eye)")
                            if head_preprocess_func:
                                ax.plot(head_preprocess_func(head_data_adjusted), label=f"{head_angle} (Head)")
                            else:
                                ax.plot(head_data_adjusted, label=f"{head_angle} (Head)")

                            # Add vertical lines for each range
                            for start, end in ranges:
                                ax.axvline(x=start, color='r', linestyle='--')
                                ax.axvline(x=end, color='r', linestyle='--')
                                ax.axvspan(start, end, color='grey', alpha=0.3)

                            ax.legend()
                            ax.set_title(f"Quaternion for P {user}, {member.split('-')[2]}, Size {size}, Pin {pin}, Auth number {num+1}")
                            ax.set_xlabel("Time")
                            ax.set_ylabel("Angle")

                        # Define and create the plot folder
                        plot_folder = os.path.join("result/", "timeseries_plots", f"{member.split('-')[0]}", "combined", "quaternion")
                        if not os.path.exists(plot_folder):
                            os.makedirs(plot_folder)

                        # Define the plot filename
                        plot_filename = f"Combined_Eye_Head_User{user}_Date{member.split('-')[2]}_Size{size}_Pin{pin}_Num{num+1}_Quaternion.png"
                        fig.savefig(os.path.join(plot_folder, plot_filename))
                        plt.close(fig)

    def head_and_smoothed_eye_quaternion_drawer(self, rotdir=None, preprocess_func=None):
        # Define angles for left eye and head
        eye_angles = ['L-QuaternionX', 'L-QuaternionY', 'L-QuaternionZ', 'L-QuaternionW']
        head_angles = ['H-QuaternionX', 'H-QuaternionY', 'H-QuaternionZ', 'H-QuaternionW']

        # Loop to process and plot data
        for member in self.studytype_users_dates:
            user = member.split('-')[1]  # Adjust according to how user is identified in your data
            for size in self.size_list:
                for pin in self.pin_list:
                    for num in range(self.default_authentications_per_person):
                        # Create a figure with subplots for each angle comparison
                        fig, axes = plt.subplots(4, 1, figsize=(10, 18))

                        # Determine the path for the specific text file
                        text_filename = f"Saccades_{member}-{size}-{pin}-{num+1}.txt"
                        text_file_path = os.path.join(rotdir, f"VRAuthStudy1-{member.split('-')[2]}/P{user}/{text_filename}")
                        # Read and parse text data from the file
                        try:
                            with open(text_file_path, 'r') as file:
                                text_data = file.read().strip()
                                # Parse the ranges from the text data
                                ranges = [list(map(int, r.split('-'))) for r in text_data.split(';') if r]
                        except FileNotFoundError:
                            ranges = []  # No ranges to add if file is not found


                        for i, (eye_angle, head_angle) in enumerate(zip(eye_angles, head_angles)):
                            # Eye data path
                            eye_filename = f"GazeCalculate_data_{member.split('-')[0]}-{user}-{member.split('-')[2]}-{str(size)}-{str(pin)}-{str(num+1)}.csv"
                            eye_file_path = os.path.join(rotdir, f"VRAuthStudy1-{member.split('-')[2]}/P{user}/{eye_filename}")

                            # Head data path
                            head_filename = f"Head_data_{member.split('-')[0]}-{user}-{member.split('-')[2]}-{str(size)}-{str(pin)}-{str(num+1)}.csv"
                            head_file_path = os.path.join(rotdir, f"VRAuthStudy1-{member.split('-')[2]}/P{user}/{head_filename}")

                            # Load eye and head data
                            eye_data = pd.read_csv(eye_file_path)[eye_angle]
                            head_data = pd.read_csv(head_file_path)[head_angle]

                            # Preprocess and adjust the angles if necessary
                            eye_data_adjusted = [x if abs(x) < 180 else (x - 360 if x > 180 else x + 360) for x in eye_data]
                            head_data_adjusted = [x if abs(x) < 180 else (x - 360 if x > 180 else x + 360) for x in head_data]

                            # smooth eye data
                            eye_data_adjusted_smoothed = smooth_data(eye_data_adjusted)

                            # Plotting each angle on the same subplot
                            ax = axes[i]
                            if preprocess_func:
                                ax.plot(preprocess_func(eye_data_adjusted_smoothed), label=f"{eye_angle} (Eye)")
                                ax.plot(preprocess_func(head_data_adjusted), label=f"{head_angle} (Head)")
                            else:
                                ax.plot(eye_data_adjusted, label=f"{eye_angle} (Eye)")
                                ax.plot(head_data_adjusted, label=f"{head_angle} (Head)")

                            # Add vertical lines for each range
                            for start, end in ranges:
                                ax.axvline(x=start, color='r', linestyle='--')
                                ax.axvline(x=end, color='r', linestyle='--')
                                ax.axvspan(start, end, color='grey', alpha=0.3)

                            ax.legend()
                            ax.set_title(f"Quaternion for User {user}, Date {member.split('-')[2]}, Size {size}, Pin {pin}, Auth number {num+1}")
                            ax.set_xlabel("Time")
                            ax.set_ylabel("Angle")

                        # Define and create the plot folder
                        plot_folder = os.path.join("result/", "timeseries_plots", f"{member.split('-')[0]}", "combined", "quaternion")
                        if not os.path.exists(plot_folder):
                            os.makedirs(plot_folder)

                        # Define the plot filename
                        plot_filename = f"Combined_Eye_Head_User{user}_Date{member.split('-')[2]}_Size{size}_Pin{pin}_Num{num+1}_Quaternion.png"
                        fig.savefig(os.path.join(plot_folder, plot_filename))
                        plt.close(fig)
    
    def time_per_size_rmanova(self, times_for_size_user):
        from statsmodels.stats.anova import AnovaRM
        # 创建一个空的DataFrame
        df = pd.DataFrame(columns=['User', 'Size', 'Pin', 'Time'])

        # 假设times_for_size是您的数据
        # 遍历每个尺寸
        for size, users_pin_times in times_for_size_user.items():
            # 遍历每个用户
            for user, pin_times in users_pin_times.items():
                # 遍历每个用户的每次时间
                for pin, times in pin_times:
                    for time in times:
                        # 为每个用户的每次尝试添加一行
                        df = df._append({'User': user, 'Size': size, 'Pin': pin, 'Time': time}, ignore_index=True)
        # 执行RMANOVA
        aovrm = AnovaRM(df, 'time', ['user', 'pin'], within=['size'])
        res = aovrm.fit()

        # 打印结果
        print(res.summary())

    def head_and_eye_spectrogram_drawer(self, rotdir=None, preprocess_func=None):
            # Define angles for left eye and head
            eye_angles = ['L-QuaternionX', 'L-QuaternionY', 'L-QuaternionZ', 'L-QuaternionW']
            head_angles = ['H-QuaternionX', 'H-QuaternionY', 'H-QuaternionZ', 'H-QuaternionW']

            # Loop to process and plot data
            for member in self.studytype_users_dates:
                user = member.split('-')[1]  # Adjust according to how user is identified in your data
                for size in self.size_list:
                    for pin in self.pin_list:
                        for num in range(self.default_authentications_per_person):
                            # Create a figure with subplots for each angle comparison
                            fig, axes = plt.subplots(4, 1, figsize=(10, 18))

                            # Determine the path for the specific text file
                            text_filename = f"Saccades_{member}-{size}-{pin}-{num+1}.txt"
                            text_file_path = os.path.join(rotdir, f"VRAuth2/P{user}/{text_filename}")
                            # Read and parse text data from the file
                            try:
                                with open(text_file_path, 'r') as file:
                                    text_data = file.read().strip()
                                    # Parse the ranges from the text data
                                    ranges = [list(map(int, r.split('-'))) for r in text_data.split(';') if r]
                            except FileNotFoundError:
                                ranges = []  # No ranges to add if file is not found


                            for i, (eye_angle, head_angle) in enumerate(zip(eye_angles, head_angles)):
                                # Eye data path
                                eye_filename = f"GazeCalculate_data_{member.split('-')[0]}-{user}-{member.split('-')[2]}-{str(size)}-{str(pin)}-{str(num+1)}.csv"
                                eye_file_path = os.path.join(rotdir, f"VRAuth2/P{user}/{eye_filename}")

                                # Head data path
                                head_filename = f"Head_data_{member.split('-')[0]}-{user}-{member.split('-')[2]}-{str(size)}-{str(pin)}-{str(num+1)}.csv"
                                head_file_path = os.path.join(rotdir, f"VRAuth2/P{user}/{head_filename}")

                                # Load eye and head data
                                eye_data = pd.read_csv(eye_file_path)[eye_angle]
                                head_data = pd.read_csv(head_file_path)[head_angle]

                                # Preprocess and adjust the angles if necessary
                                eye_data_adjusted = [x if abs(x) < 180 else (x - 360 if x > 180 else x + 360) for x in eye_data]
                                head_data_adjusted = [x if abs(x) < 180 else (x - 360 if x > 180 else x + 360) for x in head_data]


                                # smooth on hold temporary for 2024.04.29
                                # smooth eye data
                                eye_data_adjusted_smoothed = eye_data_adjusted
                                # modified 2024.04.29 original: smooth_data(eye_data_adjusted)

                                eye_data_adjusted_smoothed = np.array(eye_data_adjusted_smoothed)
                                head_data_adjusted = np.array(head_data_adjusted)

                                # Plotting each angle on the same subplot
                                ax = axes[i]
                                if preprocess_func:
                                    ax.plot(preprocess_func(eye_data_adjusted_smoothed), label=f"{eye_angle} (Eye)")
                                    ax.plot(preprocess_func(head_data_adjusted), label=f"{head_angle} (Head)")
                                else:
                                    f1, t1, Sxx1 = signal.spectrogram(eye_data_adjusted_smoothed, fs=1, nperseg=8)
                                    # f2, t2, Sxx2 = signal.spectrogram(head_data_adjusted, fs=1, nperseg=8)
                                    ax.pcolormesh(t1, f1, 10 * np.log10(Sxx1+ np.finfo(float).eps), cmap="jet", shading='auto')
                                    # ax.pcolormesh(t2, f2, 10 * np.log10(Sxx2+ np.finfo(float).eps), cmap="jet", shading='auto')
                                    # ax.plot(eye_data_adjusted, label=f"{eye_angle} (Eye)")
                                    # ax.plot(head_data_adjusted, label=f"{head_angle} (Head)")

                                # Add vertical lines for each range
                                for start, end in ranges:
                                    ax.axvline(x=start, color='r', linestyle='--')
                                    ax.axvline(x=end, color='r', linestyle='--')
                                    ax.axvspan(start, end, color='grey', alpha=0.3)

                                ax.legend()
                                # ax.set_title(f"User {user}, Date {member.split('-')[2]}, Size {size}, Pin {pin}, Auth number {num+1}")
                                ax.set_xlabel("Time")
                                ax.set_ylabel("Frequency")

                            fig.suptitle(f"User {user}, Date {member.split('-')[2]}, Size {size}, Pin {pin}, Auth number {num+1} \n " + "eye: ['L-QuaternionX', 'Y', 'Z', 'W']")
                            # Define and create the plot folder
                            plot_folder = os.path.join("result/", "timeseries_plots", f"{member.split('-')[0]}", "spectrogram")
                            if not os.path.exists(plot_folder):
                                os.makedirs(plot_folder)

                            # Define the plot filename
                            plot_filename = f"Spectrogram_Eye_User{user}_Date{member.split('-')[2]}_Size{size}_Pin{pin}_Num{num+1}.png"
                            fig.savefig(os.path.join(plot_folder, plot_filename))
                            plt.close(fig)

    def head_and_eye_FT_drawer(self, rotdir=None, preprocess_func=None):
            fs=50
            def centered_magnitude_spectrum(sig):
                return np.fft.fftshift(abs(np.fft.fft(sig)))
            # Define angles for left eye and head
            eye_angles = ['L-QuaternionX', 'L-QuaternionY', 'L-QuaternionZ', 'L-QuaternionW']
            head_angles = ['H-QuaternionX', 'H-QuaternionY', 'H-QuaternionZ', 'H-QuaternionW']

            # Loop to process and plot data
            for member in self.studytype_users_dates:
                user = member.split('-')[1]  # Adjust according to how user is identified in your data
                for size in self.size_list:
                    for pin in self.pin_list:
                        for num in range(self.default_authentications_per_person):
                            # Create a figure with subplots for each angle comparison
                            fig, axes = plt.subplots(4, 1, figsize=(10, 18))

                            # Determine the path for the specific text file
                            text_filename = f"Saccades_{member}-{size}-{pin}-{num+1}.txt"
                            text_file_path = os.path.join(rotdir, f"VRAuth2/P{user}/{text_filename}")
                            # Read and parse text data from the file
                            try:
                                with open(text_file_path, 'r') as file:
                                    text_data = file.read().strip()
                                    # Parse the ranges from the text data
                                    ranges = [list(map(int, r.split('-'))) for r in text_data.split(';') if r]
                            except FileNotFoundError:
                                ranges = []  # No ranges to add if file is not found


                            for i, (eye_angle, head_angle) in enumerate(zip(eye_angles, head_angles)):
                                # Eye data path
                                eye_filename = f"GazeCalculate_data_{member.split('-')[0]}-{user}-{member.split('-')[2]}-{str(size)}-{str(pin)}-{str(num+1)}.csv"
                                eye_file_path = os.path.join(rotdir, f"VRAuth2/P{user}/{eye_filename}")

                                # Head data path
                                head_filename = f"Head_data_{member.split('-')[0]}-{user}-{member.split('-')[2]}-{str(size)}-{str(pin)}-{str(num+1)}.csv"
                                head_file_path = os.path.join(rotdir, f"VRAuth2/P{user}/{head_filename}")

                                # Load eye and head data
                                eye_data = pd.read_csv(eye_file_path)[eye_angle]
                                head_data = pd.read_csv(head_file_path)[head_angle]

                                # Preprocess and adjust the angles if necessary
                                eye_data_adjusted = [x if abs(x) < 180 else (x - 360 if x > 180 else x + 360) for x in eye_data]
                                head_data_adjusted = [x if abs(x) < 180 else (x - 360 if x > 180 else x + 360) for x in head_data]


                                # smooth on hold temporary for 2024.04.29
                                # smooth eye data
                                eye_data_adjusted_smoothed = eye_data_adjusted
                                # modified 2024.04.29 original: smooth_data(eye_data_adjusted)

                                eye_data_adjusted_smoothed = np.array(eye_data_adjusted_smoothed)
                                head_data_adjusted = np.array(head_data_adjusted)

                                # Plotting each angle on the same subplot
                                ax = axes[i]
                                if preprocess_func:
                                    ax.plot(preprocess_func(eye_data_adjusted_smoothed), label=f"{eye_angle} (Eye)")
                                    ax.plot(preprocess_func(head_data_adjusted), label=f"{head_angle} (Head)")
                                else:
                                    freq1 = centered_magnitude_spectrum(eye_data_adjusted_smoothed)
                                    freq2 = centered_magnitude_spectrum(head_data_adjusted)
                                    ax.plot(np.linspace(-fs/2, fs/2, len(freq1)), freq1, label=f"{eye_angle} (Eye)")

                                ax.legend()
                                # ax.set_title(f"User {user}, Date {member.split('-')[2]}, Size {size}, Pin {pin}, Auth number {num+1}")
                                ax.set_ylabel("Magnitude")
                                ax.set_xlabel("Frequency")

                            fig.suptitle(f"DFT of User {user}, Date {member.split('-')[2]}, Size {size}, Pin {pin}, Auth number {num+1} \n " + "eye: ['L-QuaternionX', 'Y', 'Z', 'W']")
                            # Define and create the plot folder
                            plot_folder = os.path.join("result/", "timeseries_plots", f"{member.split('-')[0]}", "FTplot")
                            if not os.path.exists(plot_folder):
                                os.makedirs(plot_folder)

                            # Define the plot filename
                            plot_filename = f"FTplot_Eye_User{user}_Date{member.split('-')[2]}_Size{size}_Pin{pin}_Num{num+1}.png"
                            fig.savefig(os.path.join(plot_folder, plot_filename))
                            plt.close(fig)

    def _create_result_folder(self, folder_name):
        result_dir = os.path.join(os.getcwd(), "result", folder_name)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        return result_dir

    def run(self, options):
        # Check what the user wants to run and execute it
        if 'calculate_times' in options:
            average_times = self.calculate_times()
            print("Times Calculated.")

        if 'plot_time_per_size' in options:
            average_times, times_for_size, _ = self.calculate_times()
            self.plot_time_per_size(average_times, times_for_size)
            print("Plotted Time per Size.")

        if 'plot_time_per_pin_group_by_size' in options:
            average_times, _, _ = self.calculate_times()
            self.plot_time_per_pin_group_by_size(average_times)
            print("Plotted Time per Size and Pin.")

        if 'plot_time_per_size_group_by_pin' in options:
            average_times, _, _ = self.calculate_times()
            self.plot_time_per_size_group_by_pin(average_times)
            print("Plotted Time per Size and Pin.")

        if 'box_plot_time_per_size' in options:
            _, time_for_size, _ = self.calculate_times()
            self.plot_time_per_size_boxplot(time_for_size)
            print("Box Plot plotted Time per Size.")

        if 'eye_user_size_pin_num_drawer' in options:
            self.eye_user_size_pin_num_drawer(rotdir=self.rotdir)
            print("Eye Data Plotted.")

        if 'head_user_size_pin_num_drawer' in options:
            self.head_user_size_pin_num_drawer(rotdir=self.rotdir)
            print("Head Data Plotted.")

        if 'head_and_eye_drawer' in options:
            self.head_and_eye_drawer(rotdir=self.rotdir)
            print("Head and Eye Data Plotted.")

        if 'head_and_eye_quaternion_drawer' in options:
            self.head_and_eye_quaternion_drawer(rotdir=self.rotdir)
            print("Head and Eye Quaternion Data Plotted.")

        if 'head_and_smoothed_eye_quaternion_drawer' in options:
            self.head_and_smoothed_eye_quaternion_drawer(rotdir=self.rotdir)
            print("Head and Smoothed Eye Quaternion Data Plotted.")
        
        if 'time_per_size_rmanova' in options:
            _, _, times_for_size_user_pin= self.calculate_times()
            self.time_per_size_rmanova(times_for_size_user_pin)
            print("Time per Size RMANOVA Plotted.")

        if 'pin_error_rate' in options:
            self.pin_error_rate(rotdir=self.rotdir)
            print("Pin Error Rate Plotted.")

        if 'head_and_eye_spectrogram_drawer' in options:
            self.head_and_eye_spectrogram_drawer(rotdir=self.rotdir)
            print("Head and Eye Spectrogram Data Plotted.")

        if 'head_and_eye_FT_drawer' in options:
            self.head_and_eye_FT_drawer(rotdir=self.rotdir)
            print("Head and Eye FT Data Plotted.")

# rotdir是文件夹“VRAuthStudy1-1228”等存放的目录，可以是绝对目录，也可以从cwd向下获得
# Example of how to use the class with different options
drawer = Drawer(filepath="src/data_for_plot.json", size_list=[3,], pin_list=range(2,3), rotdir = os.path.join(os.getcwd(), 'data'), default_authentications_per_person=10) # todo: 从json里读取num_range
# drawer.run(options=['plot_time_per_size'])
# drawer.run(options=['plot_time_per_size_group_by_pin'])
# drawer.run(options=['plot_time_per_pin_group_by_size'])
# drawer.run(options=['box_plot_time_per_size'])
# drawer.run(options=['time_per_size_rmanova'])
# drawer.run(options=['head_and_eye_drawer'])
with open('FTplot_log.txt', 'w') as file:
    original_stdout = sys.stdout
    sys.stdout = file
    drawer.run(options=['head_and_eye_FT_drawer'])  
    sys.stdout = original_stdout
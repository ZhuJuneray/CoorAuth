import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from cycler import cycler
import os
from data_preprocess import unity_quaternion_to_euler
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


# os.chdir(os.path.join(os.getcwd(),'data'))

# ---need to modify regarding your csv file name---
user_names=["zjr","zs"]
dates="1102"
repeat_num=3


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
    for member in studytype_users_dates:
        if member.split('-')[0] == 'study1':
            for size in size_num_study1:
                for pin in pin_num:
                    for angle in ['Yaw_L', 'Pitch_L', 'Roll_L']:
                        differences = difference_gaze_head(member, size, pin, authentications_per_person, 'L', angle.split('_')[0], rotdir)
                        data[angle][(size, pin)].extend([abs(num) for num in differences])

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
            fig.savefig(os.path.join(f"result/{folder_name}", re.sub(r'["\'\[\],\s]', '', f"{stat_name}_difference_gaze_head_{angle}.png")))
            plt.close(fig)  # Close the figure to avoid displaying it

    # 绘制和保存所有统计图表
    for stat_name in stats_functions:
        draw_charts(stats[stat_name], stat_name, size_num_study1, pin_num, f"{stat_name}_difference_gaze_head")

def analyze_and_plot_time_data(filepath="src/data.json", size_list = [1, 2, 3, 4, 5, 6], pin_list = [1, 2, 3, 4]): #画每个size的用时
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
    average_times = defaultdict(dict)
    for size, pins in times.items():
        for pin, time_list in pins.items():
            if time_list:  # Ensure we don't divide by zero
                average_times[size][pin] = sum(time_list) / len(time_list)

    # Calculate the average time for each size
    size_avg_times = []
    for size in size_list:
        pin_times = [average_times[size][pin] for pin in pin_list if pin in average_times[size]]
        size_avg_time = sum(pin_times) / len(pin_times) if pin_times else 0  # Ensure not dividing by zero
        size_avg_times.append(size_avg_time)

    # Plot the bar chart and save it
    plt.figure(figsize=(10, 6))
    plt.bar(size_list, size_avg_times, color='skyblue')
    plt.xlabel('Size')
    plt.ylabel('Average Time (s)')
    plt.title('Average Time for Different Sizes')
    plt.xticks(size_list)  # Ensure every size is marked

    # Create result folder if not exist
    folder_name = os.path.join(os.getcwd(), "result")
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Save the figure
    plt.savefig(os.path.join(folder_name, "average_time_for_different_sizes.png"))
    plt.close()  # Close the plot to avoid displaying it when not needed

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


        
difference_gaze_head_size_pin_drawer(studytype_users_dates=read_data_name_from_json("src/data.json")[0], size_num_study1= [1,2,3,4,5,6], pin_num = [1,2,3,4], authentications_per_person=2, rotdir = os.path.join(os.getcwd(),'data'))
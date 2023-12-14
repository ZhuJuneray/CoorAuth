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
import re

# os.chdir(os.path.join(os.getcwd(),'data'))

# ---need to modify regarding your csv file name---
user_names=["zjr","zs"]
dates="1102"
repeat_num=3
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

#---modify end---
def expression_data_drawer(user_names, date, repeat_num):
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

def Gaze_drawer(user_names, dates, repeat_num):
    plt.rcParams['axes.prop_cycle'] = cycler(color=plt.cm.tab20.colors)
    # 
    # ---need to modify regarding your csv file name---
    plot_row1=2; plot_column1=4
    plot_row2=2; plot_column2=4
    fig_1, axs_1 = plt.subplots(plot_row1, plot_column1, figsize=(20, 10))
    fig_2, axs_2 = plt.subplots(plot_row2, plot_column2, figsize=(20, 10))
    #---modify end---

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
                      

                L_QuaternionX_raw_data=data1['L-QuaternionX']
                L_QuaternionY_raw_data=data1['L-QuaternionY']
                L_QuaternionZ_raw_data=data1['L-QuaternionZ']
                L_QuaternionW_raw_data=data1['L-QuaternionW']
                R_QuaternionX_raw_data=data1['R-QuaternionX']
                R_QuaternionY_raw_data=data1['R-QuaternionY']
                R_QuaternionZ_raw_data=data1['R-QuaternionZ']
                R_QuaternionW_raw_data=data1['R-QuaternionW']

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

def Head_data_drawer(user_names, dates, repeat_num):

    plt.rcParams['axes.prop_cycle'] = cycler(color=plt.cm.tab20.colors)
    plot_row=3; plot_column=3
    fig_1, axs_1 = plt.subplots(plot_row, plot_column, figsize=(20, 10))
    fig_2, axs_2 = plt.subplots(3, 4, figsize=(20, 10))
    #---modify end---
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

def unity_angle_drawer(user_names, dates, repeat_num, curve_num_per_fig=3, source='Gaze'):
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
                    data = pd.read_csv(os.path.join("data/unity_processed_data","GazeCalculate_data_" + user + "-" + date + "-" + str(num+1) + "_unity_processed.csv"))
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
                    data = pd.read_csv(os.path.join("data/unity_processed_data","Head_data_" + user + "-" + date + "-" + str(num+1) + "_unity_processed.csv"))
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

def difference_gaze_head_drawer(user_names, dates, repeat_num, curve_num_per_fig=3): # num从1开始
    fig_num=0
    user_plotted=[]
    plot_row=2; plot_column=3
    fig, axs = plt.subplots(plot_row, plot_column, figsize=(20, 10))
    for user in user_names:
        for date in dates:
            for num in range(repeat_num):
                if fig_num == 0:
                    fig, axs = plt.subplots(plot_row, plot_column, figsize=(20, 10))
                fig_num+=1
                user_plotted.append(str(user) + str(num+1))
                Yaw_L = difference_gaze_head(user, date, num+1, eye='L', angle='Yaw')
                Pitch_L = difference_gaze_head(user, date, num+1, eye='L', angle='Pitch')
                Roll_L = difference_gaze_head(user, date, num+1, eye='L', angle='Roll')
                Yaw_R = difference_gaze_head(user, date, num+1, eye='R', angle='Yaw')
                Pitch_R = difference_gaze_head(user, date, num+1, eye='R', angle='Pitch')
                Roll_R = difference_gaze_head(user, date, num+1, eye='R', angle='Roll')
    
                axs[0,0].plot(Yaw_L, label=user+str(num+1)+'_'+date)
                axs[0,0].set_title('L_Gaze_Yaw-Head_Yaw')
                axs[0,1].plot(Pitch_L, label=user+str(num+1)+'_'+date)
                axs[0,1].set_title('L_Gaze_Pitch-Head_Pitch')
                axs[0,2].plot(Roll_L, label=user+str(num+1)+'_'+date)
                axs[0,2].set_title('L_Gaze_Roll-Head_Roll')
                axs[1,0].plot(Yaw_R, label=user+str(num+1)+'_'+date)
                axs[1,0].set_title('R_Gaze_Yaw-Head_Yaw')
                axs[1,1].plot(Pitch_R, label=user+str(num+1)+'_'+date)
                axs[1,1].set_title('R_Gaze_Pitch-Head_Pitch')
                axs[1,2].plot(Roll_R, label=user+str(num+1)+'_'+date)
                axs[1,2].set_title('R_Gaze_Roll-Head_Roll')
                if fig_num == curve_num_per_fig:
                    for i in range(plot_row):
                        for j in range(plot_column):
                            axs[i, j].legend()
                    fig.savefig(os.path.join("result/difference_gaze_head",re.sub(r'["\'\[\],\s]', '', "difference_gaze_head" + str(user_plotted) + ".png")))
                    fig_num=0
                    user_plotted=[]
                    plt.clf()
    
    
def difference_gaze_lr_euler_angle_drawer(user_names, dates, repeat_num, curve_num_per_fig=3):
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

unity_angle_drawer(['zjr'],['1118'],6,source='Head')

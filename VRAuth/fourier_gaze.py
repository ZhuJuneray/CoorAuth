import pandas as pd
import matplotlib.pyplot as plt
import  os
import numpy as np
os.chdir(os.path.join(os.getcwd(),'VRAuth'))


user_names=['jyc']
dates=['1111']
repeat_num=2
plot_row1=2; plot_column1=3
plot_row2=2; plot_column2=3
# fig_1, axs_1 = plt.subplots(plot_row1, plot_column1, figsize=(20, 10))
# fig_2, axs_2 = plt.subplots(plot_row2, plot_column2, figsize=(20, 10))

diagram_num = 0
for user in user_names:
    
    for num in [0]:
        fig_2, axs_2 = plt.subplots(plot_row2, plot_column2, figsize=(20, 10))
        for date in dates:
            data1 = pd.read_csv(os.path.join("unity_processed_data","GazeCalculate_data_" + user + "-" + date + "-" + str(num+1) + "_unity_processed.csv"))
           
            # Create DataFrame
            df1 = pd.DataFrame(data1)

            # Plotting
            # plt.figure(figsize=(12, 6))

            slice_l=100
            slice_r=300

            L_Yaw_zero = df1['L_Yaw'][0]
            L_Yaw_df = [x - L_Yaw_zero if abs(x - L_Yaw_zero) < 200 else (x - L_Yaw_zero -  360 if x- L_Yaw_zero >200 else x - L_Yaw_zero + 360 ) for x in df1['L_Yaw']]
            L_Yaw_df_slice = L_Yaw_df[slice_l:slice_r]
            fft_L_Yaw = np.fft.fft(L_Yaw_df)
            freq_L_Yaw=np.fft.fftfreq(len(L_Yaw_df),0.02)

            
            # print('L_Yaw_FFT: ',np.abs(fft_L_Yaw))
            axs_2[0, 0].plot(L_Yaw_df, label='Gaze: '+user+str(num+1)+'_'+date)
            axs_2[0, 0].set_title('L_Gaze_Yaw')
            axs_2[0, 0].axvline(x=slice_l, color='r')
            axs_2[0, 0].axvline(x=slice_r, color='r')
            axs_2[1, 0].plot(freq_L_Yaw, np.abs(fft_L_Yaw), label='Gaze: '+user+str(num+1)+'_'+date)
            axs_2[1, 0].set_title('L_Gaze_Yaw_FFT')

            axs_2[0, 1].plot(L_Yaw_df_slice, color= 'r')
            axs_2[0, 1].set_title(f'L_Gaze_Yaw_slice_red[{slice_l}:{slice_r}]')
            fft_L_Yaw = np.fft.fft(L_Yaw_df_slice)
            freq_L_Yaw=np.fft.fftfreq(len(L_Yaw_df_slice),0.02)
            axs_2[1, 1].plot(freq_L_Yaw, np.abs(fft_L_Yaw), label='Gaze: '+user+str(num+1)+'_'+date)
            axs_2[1, 1].set_title('L_Gaze_Yaw__slice_red_FFT')
            axs_2[1, 1].set_xlim(0, 10)
            axs_2[1, 1].set_ylim(0, 1200)
            axs_2[1, 1].set_xlabel('Frequency(Hz)')
            axs_2[1, 1].set_ylabel('Amplitude')


            execute_block = True # skip specific line drawing: True for skipping and False for drawing all line  
            if execute_block:
                slice_l=305
                slice_r=350
                L_Yaw_df_slice = L_Yaw_df[slice_l:slice_r]
                fft_L_Yaw = np.fft.fft(L_Yaw_df_slice)
                freq_L_Yaw=np.fft.fftfreq(len(L_Yaw_df_slice),0.02)
                axs_2[0, 0].axvline(x=slice_l, color='g')
                axs_2[0, 0].axvline(x=slice_r, color='g')
                axs_2[0, 2].plot(L_Yaw_df_slice, color= 'g')
                axs_2[0, 2].set_title(f'L_Gaze_Yaw_slice_green[{slice_l}:{slice_r}]')
                axs_2[1, 2].plot(freq_L_Yaw, np.abs(fft_L_Yaw), label='Gaze: '+user+str(num+1)+'_'+date)
                axs_2[1, 2].set_title('L_Gaze_Yaw__slice_green_FFT')
                axs_2[1, 2].set_xlim(0, 10)
                axs_2[1, 2].set_ylim(0, 500)
                axs_2[1, 2].set_xlabel('Frequency(Hz)')
                axs_2[1, 2].set_ylabel('Amplitude')


            # axs_2[1, 1].set_xlim(0, 10)
            # axs_2[1, 1].set_ylim(0, 500)
            
            # L_Pitch_zero = df1['L_Pitch'][0] if df1['L_Pitch'][0] < 180 else df1['L_Pitch'][0] - 360
            # axs_2[0, 1].plot([x -L_Pitch_zero if x < 180 else x - L_Pitch_zero - 360 for x in df1['L_Pitch']])
           
            # axs_2[num, 1].set_title('L_Gaze_Pitch & Head_Pitch')
            # L_Roll_zero = df1['L_Roll'][0] if df1['L_Roll'][0] < 180 else df1['L_Roll'][0] - 360
            # axs_2[num, 2].plot([x - L_Roll_zero if x < 180 else x  - L_Roll_zero - 360 for x in df1['L_Roll']])
            
            # axs_2[num, 2].set_title('L_Gaze_Roll & Head_Roll')
            # axs_2[1, 0].plot(df['R_Yaw'])
            # axs_2[1, 0].set_title('R_Yaw')
            # axs_2[1, 1].plot([x  if x < 180 else x - 360 for x in df['R_Pitch']])
            # axs_2[1, 1].set_title('R_Pitch')
            # axs_2[1, 2].plot([x  if x < 180 else x - 360 for x in df['R_Roll']])
            # axs_2[1, 2].set_title('R_Roll')

            axs_2[0, 0].legend()
            axs_2[1, 0].legend()
        fig_2.savefig(os.path.join("unity_processed_picture","FFT_Gaze_angle['" + str(user) +' ' + str(num+1) + "'].png"))
        plt.clf()
    diagram_num += 1
    
    

# # Plot each column
# for column in df.columns:
#     plt.plot(df[column], label=column)

# plt.title("Euler Angles over Time")
# plt.xlabel("Sample Index")
# plt.ylabel("Angle (degrees)")
# plt.legend()
# plt.tight_layout()
# fig_1.savefig(os.path.join("/unity_processed_picture",'Gaze_data_raw' + str(user_names) + '.png'))
# fig_2.savefig(os.path.join("unity_processed_picture",'difference_euler_angle_gaze_head' + str(user_names) + '.png'))
# plt.grid(True)
# plt.show()

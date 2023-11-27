import pandas as pd
import matplotlib.pyplot as plt
import  os
os.chdir(os.path.join(os.getcwd(),'VRAuth'))


user_names=['zjr','yf','jyc','wgh','lhy']
dates=['1111']
repeat_num=2
plot_row1=2; plot_column1=3
plot_row2=2; plot_column2=3
# fig_1, axs_1 = plt.subplots(plot_row1, plot_column1, figsize=(20, 10))
# fig_2, axs_2 = plt.subplots(plot_row2, plot_column2, figsize=(20, 10))

diagram_num = 0
for user in user_names:
    fig_2, axs_2 = plt.subplots(plot_row2, plot_column2, figsize=(20, 10))
    for num in range(repeat_num):
        for date in dates:
            data1 = pd.read_csv(os.path.join("unity_processed_data","GazeCalculate_data_" + user + "-" + date + "-" + str(num+1) + "_unity_processed.csv"))
            data2 = pd.read_csv(os.path.join("unity_processed_data","Head_data_" + user + "-" + date + "-" + str(num+1) + "_unity_processed.csv"))
            # Create DataFrame
            df1 = pd.DataFrame(data1)
            df2 = pd.DataFrame(data2)

            # Plotting
            # plt.figure(figsize=(12, 6))

            L_Yaw_zero = df1['L_Yaw'][0]
            L_Yaw_df = [x - L_Yaw_zero if abs(x - L_Yaw_zero) < 200 else (x - L_Yaw_zero -  360 if x- L_Yaw_zero >200 else x - L_Yaw_zero + 360 ) for x in df1['L_Yaw']]
            axs_2[num, 0].plot(L_Yaw_df, label='Gaze: '+user+str(num+1)+'_'+date)
            Yaw_zero = df2['Yaw'][0] if df2['Yaw'][0] < 180 else df2['Yaw'][0] - 360
            axs_2[num, 0].plot([x - Yaw_zero if x < 180 else x - Yaw_zero -  360 for x in df2['Yaw']], label='Head: '+user+str(num+1)+'_'+date)
            axs_2[num, 0].set_title('L_Gaze_Yaw & Head_Yaw')
            L_Pitch_zero = df1['L_Pitch'][0] if df1['L_Pitch'][0] < 180 else df1['L_Pitch'][0] - 360
            axs_2[num, 1].plot([x -L_Pitch_zero if x < 180 else x - L_Pitch_zero - 360 for x in df1['L_Pitch']])
            Pitch_zero = df2['Pitch'][0] if df2['Pitch'][0] < 180 else df2['Pitch'][0] - 360
            axs_2[num, 1].plot([x - Pitch_zero if x < 180 else x - Pitch_zero - 360 for x in df2['Pitch']])
            axs_2[num, 1].set_title('L_Gaze_Pitch & Head_Pitch')
            L_Roll_zero = df1['L_Roll'][0] if df1['L_Roll'][0] < 180 else df1['L_Roll'][0] - 360
            axs_2[num, 2].plot([x - L_Roll_zero if x < 180 else x  - L_Roll_zero - 360 for x in df1['L_Roll']])
            Roll_zero = df2['Roll'][0] if df2['Roll'][0] < 180 else df2['Roll'][0] - 360
            axs_2[num, 2].plot([x - Roll_zero if x < 180 else x - Roll_zero - 360 for x in df2['Roll']])
            axs_2[num, 2].set_title('L_Gaze_Roll & Head_Roll')
            # axs_2[1, 0].plot(df['R_Yaw'])
            # axs_2[1, 0].set_title('R_Yaw')
            # axs_2[1, 1].plot([x  if x < 180 else x - 360 for x in df['R_Pitch']])
            # axs_2[1, 1].set_title('R_Pitch')
            # axs_2[1, 2].plot([x  if x < 180 else x - 360 for x in df['R_Roll']])
            # axs_2[1, 2].set_title('R_Roll')

            axs_2[0, 0].legend()
            axs_2[1, 0].legend()
    
    diagram_num += 1
    fig_2.savefig(os.path.join("unity_processed_picture","difference_euler_angle_gaze_head['" + str(user) + "'].png"))
    plt.clf()

# # Plot each column
# for column in df.columns:
#     plt.plot(df[column], label=column)

# plt.title("Euler Angles over Time")
# plt.xlabel("Sample Index")
plt.ylabel("Angle (degrees)")
plt.legend()
plt.tight_layout()
# fig_1.savefig(os.path.join("/unity_processed_picture",'Gaze_data_raw' + str(user_names) + '.png'))
# fig_2.savefig(os.path.join("unity_processed_picture",'difference_euler_angle_gaze_head' + str(user_names) + '.png'))
# plt.grid(True)
# plt.show()

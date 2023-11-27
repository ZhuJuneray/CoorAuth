import pandas as pd
import matplotlib.pyplot as plt
import  os
os.chdir(os.path.join(os.getcwd(),'VRAuth'))


user_names=['jyc']
dates=['1111']
repeat_num=2
plot_row1=2; plot_column1=3
plot_row2=1; plot_column2=3
# fig_1, axs_1 = plt.subplots(plot_row1, plot_column1, figsize=(20, 10))
fig_2, axs_2 = plt.subplots(plot_row2, plot_column2, figsize=(20, 10))


for user in user_names:
    for num in range(repeat_num):
        for date in dates:
            data = pd.read_csv(os.path.join("unity_processed_data","GazeCalculate_data_" + user + "-" + date + "-" + str(num+1) + "_unity_processed.csv"))

            # Create DataFrame
            df = pd.DataFrame(data)

            # Plotting
            # plt.figure(figsize=(12, 6))

            axs_2[0].plot([x - y if abs(x - y) < 180 else (x - y - 360 if x - y >180 else x - y +360) for x, y in zip(df['L_Yaw'], df['R_Yaw'])], label=user+str(num+1)+'_'+date)
            axs_2[0].set_title('L_Yaw-R_Yaw')
            axs_2[1].plot([x - y if abs(x - y) < 180 else (x - y - 360 if x - y >180 else x - y +360) for x, y in zip(df['L_Pitch'], df['R_Pitch'])], label=user+str(num+1)+'_'+date)
            axs_2[1].set_title('L_Pitch-R_Pitch')
            axs_2[2].plot([x - y if abs(x - y) < 180 else (x - y - 360 if x - y >180 else x - y +360) for x, y in zip(df['L_Roll'], df['R_Roll'])], label=user+str(num+1)+'_'+date)
            axs_2[2].set_title('L_Roll-R_Roll')



            # axs_2[0, 1].plot([x  if x < 180 else x - 360 for x in df['Pitch']])
            # axs_2[0, 1].set_title('Pitch')
            # axs_2[0, 2].plot([x  if x < 180 else x - 360 for x in df['Roll']])
            # axs_2[0, 2].set_title('Roll')
            # axs_2[1, 0].plot(df['R_Yaw'])
            # axs_2[1, 0].set_title('R_Yaw')
            # axs_2[1, 1].plot([x  if x < 180 else x - 360 for x in df['R_Pitch']])
            # axs_2[1, 1].set_title('R_Pitch')
            # axs_2[1, 2].plot([x  if x < 180 else x - 360 for x in df['R_Roll']])
            # axs_2[1, 2].set_title('R_Roll')

            axs_2[0].legend()

# # Plot each column
# for column in df.columns:
#     plt.plot(df[column], label=column)

# plt.title("Euler Angles over Time")
# plt.xlabel("Sample Index")
plt.ylabel("Angle (degrees)")
plt.legend()
plt.tight_layout()
# fig_1.savefig(os.path.join("/unity_processed_picture",'Gaze_data_raw' + str(user_names) + '.png'))
fig_2.savefig(os.path.join("unity_processed_picture",'Gaze_LR_euler_angle_difference_by_unity' + str(user_names) + '.png'))
# plt.grid(True)
plt.show()

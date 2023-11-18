"""
Author: June Ray
Email: juneray2003@gmail.com
GitHub: https://github.com/ZhuJunray/
Version: 1.0.0
Date: 2023-11-08
Description: This is a Python script to draw csv data from Quest.
"""
import pandas as pd
import matplotlib.pyplot as plt

# ---need to modify regarding your csv file name---
user_names=["zs"]
dates=["1109","1108"]
repeat_num=1
fig, axs = plt.subplots(4, 2, figsize=(20, 10))
#---modify end---

for user in user_names:
    for num in range(repeat_num):
        for date in dates:

            execute_block = False # skip specific line drawing: True for skipping and False for drawing all line
            if execute_block:
                if num==2 and user=='zjr':
                    continue 

            data=pd.read_csv("Head_data_"+user+'_'+date+'_'+str(num)+'.csv')
            print(data.index)

            Vector3X_data=data['Vector3X']
            Vector3Y_data=data['Vector3Y']
            Vector3Z_data=data['Vector3Z']
            QuaternionX_data=data['QuaternionX']
            QuaternionY_data=data['QuaternionY']
            QuaternionZ_data=data['QuaternionZ']
            QuaternionW_data=data['QuaternionW']
 
            print(Vector3X_data[0])

        # axs[0, 0].plot(Vector3X_data-Vector3X_data[0],label=user+str(num+1))
        # axs[0, 0].set_title('Vector3X')
        # axs[0, 1].plot(Vector3Y_data-Vector3Y_data[0],label=user+str(num+1))
        # axs[0, 1].set_title('Vector3Y_data')
        # axs[1, 0].plot(Vector3Z_data-Vector3Z_data[0],label=user+str(num+1))
        # axs[1, 0].set_title('Vector3Z')
        # axs[1, 1].plot(QuaternionX_data-QuaternionX_data[0],label=user+str(num+1))
        # axs[1, 1].set_title('QuaternionX')
        # axs[2, 0].plot(QuaternionY_data-QuaternionY_data[0],label=user+str(num+1))
        # axs[2, 0].set_title('QuaternionY')
        # axs[2, 1].plot(QuaternionZ_data-QuaternionZ_data[0],label=user+str(num+1))
        # axs[2, 1].set_title('QuaternionZ')
        # axs[3, 0].plot(QuaternionW_data-QuaternionW_data[0],label=user+str(num+1))
        # axs[3, 0].set_title('QuaternionW')
        # for i in range(4):
        #     for j in range(2):
        #         axs[i, j].legend()

plt.tight_layout()
plt.savefig('Head_data_drawed.png')
plt.show()

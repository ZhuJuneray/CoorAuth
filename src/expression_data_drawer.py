"""
Author: June Ray
Email: juneray2003@gmail.com
GitHub: https://github.com/ZhuJunray/
Version: 1.0.0
Date: 2023-11-02
Description: This is a Python script to draw csv data from Quest.
"""
import pandas as pd
import matplotlib.pyplot as plt

# ---need to modify regarding your csv file name---
user_names=["zjr","zs"]
date="1102"
repeat_num=3
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
#---modify end---

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

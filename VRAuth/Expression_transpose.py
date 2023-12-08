import pandas as pd
import os
import glob

def remove_csv_extension(filename):
    # 检查字符串是否以".csv"结尾
    if filename.endswith('.csv'):
        # 从字符串中删除最后四个字符(".csv")
        return filename[:-4]
    else:
        # 如果字符串不是以".csv"结尾，返回原始字符串
        return filename

# print(os.getcwd())
os.chdir(os.path.join(os.getcwd(),'VRAuth 2'))
files = glob.glob('Expression*[0-9].csv')

for file in files:
    df = pd.read_csv(file,header=None,skiprows=1)
    df=df.drop(columns=[0])
    transposed_df = df.transpose()
    if os.path.exists(os.path.join(os.getcwd(),'Expression_data'))==False:
        os.mkdir(os.path.join(os.getcwd(),'Expression_data'))
    transposed_df.to_csv(os.path.join('Expression_data',remove_csv_extension(file) + '-transposed.csv'), header=['NoseWrinklerR','CheekRaiserR','LidTightenerR','UpperLipRaiserR','EyesClosedR','UpperLidRaiserR'], index=False)

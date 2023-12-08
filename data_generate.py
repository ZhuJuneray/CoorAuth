import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import  os
os.chdir(os.path.join(os.getcwd(),'VRAuth2'))

def difference_yaw_gaze_head(user,date,num):
    data1 = pd.read_csv(os.path.join("unity_processed_data","GazeCalculate_data_" + user + "-" + date + "-" + str(num+1) + "_unity_processed.csv"))
    data2 = pd.read_csv(os.path.join("unity_processed_data","Head_data_" + user + "-" + date + "-" + str(num+1) + "_unity_processed.csv"))
    # Create DataFrame
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)

    L_Yaw_zero = df1['L_Yaw'][0]
    L_Yaw_df = [x - L_Yaw_zero if abs(x - L_Yaw_zero) < 200 else (x - L_Yaw_zero -  360 if x- L_Yaw_zero >200 else x - L_Yaw_zero + 360 ) for x in df1['L_Yaw']]
    Yaw_zero = df2['Yaw'][0] if df2['Yaw'][0] < 180 else df2['Yaw'][0] - 360
    Yaw_df = [x - Yaw_zero if x < 180 else x - Yaw_zero -  360 for x in df2['Yaw']]

    return [x - y for x, y in zip(L_Yaw_df, Yaw_df)]

def euler_angle(user, date, num):
    data = pd.read_csv(os.path.join("unity_processed_data","GazeCalculate_data_" + user + "-" + date + "-" + str(num+1) + "_unity_processed.csv"))
    # Create DataFrame
    df = pd.DataFrame(data)

    L_R_Yaw = [x - y if abs(x - y) < 180 else (x - y - 360 if x - y >180 else x - y +360) for x, y in zip(df['L_Yaw'], df['R_Yaw'])]
    L_R_Pitch = [x - y if abs(x - y) < 180 else (x - y - 360 if x - y >180 else x - y +360) for x, y in zip(df['L_Pitch'], df['R_Pitch'])]
    L_R_Roll = [x - y if abs(x - y) < 180 else (x - y - 360 if x - y >180 else x - y +360) for x, y in zip(df['L_Roll'], df['R_Roll'])]
    return L_R_Yaw, L_R_Pitch, L_R_Roll

def smooth_data(arr, window_parameter=31, polyorder_parameter=2):
    arr_smoothed = savgol_filter(arr, window_length=window_parameter, polyorder=polyorder_parameter)
    return arr_smoothed


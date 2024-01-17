import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

class HeadAngleDrawer:
    def __init__(self, studytype_users_dates, size_num_study1, pin_num, authentications_per_person, rotdir):
        self.studytype_users_dates = studytype_users_dates
        self.size_num_study1 = size_num_study1
        self.pin_num = pin_num
        self.authentications_per_person = authentications_per_person
        self.rotdir = rotdir
        self.data = self.initialize_data()
        self.stats = None

    def initialize_data(self):
        # Initialize dictionary to store data
        return {angle: {(size, pin): [] for size in self.size_num_study1 for pin in self.pin_num}
                for angle in ['Yaw', 'Pitch', 'Roll']}

    def collect_data(self):
        # Collect data for head angles
        for member in self.studytype_users_dates:
            if member.split('-')[0] == 'study1':
                for size in self.size_num_study1:
                    for pin in self.pin_num:
                        for angle in ['Yaw', 'Pitch', 'Roll']:
                            # Replace the function below with the actual data collection for head angles
                            angles = pd.read_csv(os.path.join(rotdir, f"data{member.split('-')[2]}/P{member.split('-')[1]}/Head_data_{member.split('-')[0]}-{member.split('-')[1]}-{member.split('-')[2]}-{str(size)}-{str(pin)}-{str(num)}_unity_processed.csv"))
                            your_function_to_collect_head_angles(member, size, pin, self.authentications_per_person, angle, self.rotdir)
                            self.data[angle][(size, pin)].extend(angles)

    def apply_statistics(self):
        # Define statistical functions
        stats_functions = {
            'mean': np.nanmean,
            'max': np.nanmax,
            'min': np.nanmin,
            'var': np.nanvar,
            'median': np.nanmedian,
            'rms': lambda x: np.sqrt(np.nanmean(np.square(x)))  # Root Mean Square
        }
        
        # Apply statistical functions to the data
        self.stats = {stat: {angle: {key: func(values) for key, values in data_angle.items()}
                             for angle, data_angle in self.data.items()}
                      for stat, func in stats_functions.items()}

    def draw_charts(self, stat_data, stat_name, folder_name):
        # Plotting charts for each statistical measure
        colors = cm.viridis(np.linspace(0, 1, len(self.pin_num)))
        n_sizes = len(self.size_num_study1)
        
        # Create directory for result
        if not os.path.exists(f"result/{folder_name}"):
            os.makedirs(f"result/{folder_name}")
        
        # Plot each angle's chart
        for angle in stat_data:
            fig, axs = plt.subplots(1, n_sizes, figsize=(n_sizes * 5, 6), squeeze=False)
            for i, size in enumerate(self.size_num_study1):
                for j, pin in enumerate(self.pin_num):
                    value = stat_data[angle].get((size, pin), 0)
                    axs[0, i].bar(pin, value, color=colors[j], label=f'Pin {pin}')
                axs[0, i].set_title(f'Size {size}')
                axs[0, i].set_xlabel('Pin Number')
                axs[0, i].set_ylabel(f'{stat_name.capitalize()} of {angle}')
                axs[0, i].legend()
            plt.tight_layout()
            fig.savefig(os.path.join(f"result/{folder_name}", re.sub(r'["\'\[\],\s]', '', f"{stat_name}_{angle}.png")))
            plt.close(fig)

    def plot_head_angles(self):
        # Collect data
        self.collect_data()
        
        # Apply statistics
        self.apply_statistics()
        
        # Draw and save all statistical charts
        for stat_name in self.stats:
            self.draw_charts(self.stats[stat_name], stat_name, f"{stat_name}_head_angles")

from sklearn.feature_selection import SelectKBest, f_classif

# X = np.array([[1,1,1,1,0,2,3],
#               [1,1,1,1,0,2,3],
#               [1,1,1,1,0,2,3],
#               [1,1,1,1,0,2,3],
#               [1,1,1,1,0,2,3],
#               [1,1,1,1,3,2,3],
#               [1,1,1,1,3,2,3],
#               [1,1,1,1,3,2,3],
#               [1,1,1,1,3,2,3],
#               [1,1,1,1,3,2,3]])

# y = np.array([0,0,0,0,0,1,1,1,1,1])

# selector = SelectKBest(score_func=f_classif, k='all')  # k='all'意味着选择所有特征
# selector.fit(X, y)

# # 获取各特征的Fisher得分
# scores = selector.scores_

# # 打印特征得分
# print("Feature scores:", scores)


# from sklearn.svm import SVC
# from sklearn.feature_selection import RFE

# # Initialize SVM classifier
# svc = SVC(kernel="linear")

# # Select number of features you want to retain. For example, let's keep 10 features
# num_features_to_select = 10

# # Initialize RFE with the linear SVM classifier
# rfe = RFE(estimator=svc, n_features_to_select=num_features_to_select)

# # Fit RFE
# rfe = rfe.fit(X, y)

# # Print the ranking of features
# ranking = rfe.ranking_
# print('Feature Ranking: %s' % ranking)
# important_feature_indices = np.where(ranking == 1)[0]
# print("Indices of the most important features:", important_feature_indices)

# if np.isinf(a).any():
#     print(f"inf indices:{np.where(np.isinf(a))}")
# print(np.argsort(a))
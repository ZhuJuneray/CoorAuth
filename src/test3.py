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

import os
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})
def head_and_eye_drawer(studytype_users_dates, size_list, pin_list, default_authentications_per_person, rotdir=None, preprocess_func=None):
        # Define angles for left eye and head
        eye_angles = ['L_Yaw', 'L_Pitch', 'L_Roll']
        head_angles = ['Yaw', 'Pitch', 'Roll']
        sbuplot_titles = ['(a) Yaw', '(b) Pitch', '(c) Roll']

        # Loop to process and plot data
        for member in studytype_users_dates:
            user = member.split('-')[1]  # Adjust according to how user is identified in your data
            for size in size_list:
                for pin in pin_list:
                    for num in range(default_authentications_per_person):
                        
                        # Create a figure with subplots for each angle comparison
                        # fig, axes = plt.subplots(3, 1, figsize=(10, 15))
                        fig, axes = plt.subplots(1, 3, figsize=(30, 5))
                        # 为底部标题调整子图布局
                        fig.subplots_adjust(bottom=0.25)

                        # Determine the path for the specific text file
                        text_filename = f"Saccades_{member}-{size}-{pin}-{num+1}.txt"
                        text_file_path = os.path.join(rotdir, f"VRAuthStudy1-{member.split('-')[2]}/P{user}/{text_filename}")
                        # Read and parse text data from the file
                        try:
                            with open(text_file_path, 'r') as file:
                                text_data = file.read().strip()
                                # Parse the ranges from the text data
                                ranges = [list(map(int, r.split('-'))) for r in text_data.split(';') if r]
                        except FileNotFoundError:
                            ranges = []  # No ranges to add if file is not found
                        
                        print(f"ranges: {ranges}")
                        ranges = [[3, 18], [43, 65], [91, 108], [131, 144]]


                        for i, (eye_angle, head_angle) in enumerate(zip(eye_angles, head_angles)):
                            # Eye data path
                            eye_filename = f"GazeCalculate_data_{member.split('-')[0]}-{user}-{member.split('-')[2]}-{str(size)}-{str(pin)}-{str(num+1)}.csv"
                            eye_file_path = os.path.join(rotdir, f"VRAuthStudy1Angle-{member.split('-')[2]}/P{user}/{eye_filename}")

                            # Head data path
                            head_filename = f"Head_data_{member.split('-')[0]}-{user}-{member.split('-')[2]}-{str(size)}-{str(pin)}-{str(num+1)}.csv"
                            head_file_path = os.path.join(rotdir, f"VRAuthStudy1Angle-{member.split('-')[2]}/P{user}/{head_filename}")

                            # Load eye and head data
                            eye_data = pd.read_csv(eye_file_path)[eye_angle]
                            head_data = pd.read_csv(head_file_path)[head_angle]

                            # Preprocess and adjust the angles if necessary
                            eye_data_adjusted = [x if abs(x) < 180 else (x - 360 if x > 180 else x + 360) for x in eye_data]
                            head_data_adjusted = [x if abs(x) < 180 else (x - 360 if x > 180 else x + 360) for x in head_data]

                            # Plotting each angle on the same subplot
                            ax = axes[i]
                            if preprocess_func:
                                ax.plot(preprocess_func(eye_data_adjusted))
                                ax.plot(preprocess_func(head_data_adjusted))
                            else:
                                if i ==2: 
                                    ax.plot(eye_data_adjusted, label=f"Eye Gaze")
                                    ax.plot(head_data_adjusted, label=f"Head")
                                else:
                                    ax.plot(eye_data_adjusted)
                                    ax.plot(head_data_adjusted)

                            # Add vertical lines for each range
                            for start, end in ranges:
                                ax.axvline(x=start, color='r', linestyle='--')
                                ax.axvline(x=end, color='r', linestyle='--')
                                ax.axvspan(start, end, color='grey', alpha=0.3)

                            x_ticks = ax.get_xticks()
                            ax.set_xticklabels([f"{x*0.02:.2f}" for x in x_ticks])
                            ax.set_yticklabels(ax.get_yticks())

                            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                            # ax.set_title(f"Euler Angle for User {user}, Date {member.split('-')[2]}, Size {int(size)-1}, Pin {pin}, Auth number {num+1}")
                            ax.set_xlabel(f"Time(s)\n{sbuplot_titles[i]}")
                            ax.set_ylabel("Angle($^\circ$)")
                            # Remove right and top spines/borders
                            ax.spines['right'].set_visible(False)
                            ax.spines['top'].set_visible(False)
                        
                        # fig.text(0.5, 0.05, 'Time series data for participant P14 drawing pattern No.13, Under Size S2', ha='center')
                        # fig.tight_layout()
                        

                        # Define and create the plot folder
                        plot_folder = os.path.join(os.getcwd(), "result", "study1")
                        if not os.path.exists(plot_folder):
                            os.makedirs(plot_folder)

                        # Define the plot filename
                        plot_filename = f"time_series_example_plot_{num}.png"
                        
                        fig.savefig(os.path.join(plot_folder, plot_filename))
                        plt.close(fig)

# Define the studytype, users, and dates
studytype_users_dates = ['study2-14-1231']
size_list = ['3']
pin_list = ['13']
default_authentications_per_person = 1
rotdir = os.path.join(os.getcwd(), 'data')
head_and_eye_drawer(studytype_users_dates, size_list, pin_list, default_authentications_per_person, rotdir=rotdir)
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import gaussian_kde

# (size, pin) to target distance mapping
target_distance_mapping = {
    (3, 0): 30,
    (3, 1): 22.2,
    (3, 2): 30,
    (3, 3): 22.2,
    (3, 4): 0,
    (3, 5): 22.2,
    (3, 6): 30,
    (3, 7): 22.2,
    (3, 8): 30,
}

# Get fixation ranges from saccades data
def get_fixation_ranges(saccades_ranges, total_length):
    saccades = [tuple(map(int, item.split('-'))) for item in saccades_ranges.split(';') if item]
    fixation_ranges = []
    current_start = 0
    for sac_start, sac_end in saccades:
        if current_start < sac_start:
            fixation_ranges.append((current_start, sac_start - 1))
        current_start = sac_end + 1
    if current_start < total_length:
        fixation_ranges.append((current_start, total_length - 1))
    return fixation_ranges

# Calculate THM
def calculate_thm(df):
    thms = []
    for index, row in df.iterrows():
        yaw = np.abs(np.degrees(np.arctan2(2 * (row['H-QuaternionW'] * row['H-QuaternionZ'] + row['H-QuaternionX'] * row['H-QuaternionY']),
                                           1 - 2 * (row['H-QuaternionY']**2 + row['H-QuaternionZ']**2))))
        pitch = np.abs(np.degrees(np.arcsin(2 * (row['H-QuaternionW'] * row['H-QuaternionY'] - row['H-QuaternionZ'] * row['H-QuaternionX']))))
        thm = np.arccos(1 - 2 * (np.sin(np.radians(yaw / 2))**2 + np.sin(np.radians(pitch / 2))**2))
        thms.append(np.degrees(np.abs(thm)))
    return thms

# Data directory
parent_dir = "data/VRAuth2"  # Replace with your directory path

thm_data = {}  # Store THM values for each participant
participants = []

# Iterate over the parent directory to process each subfolder
for subdir in os.listdir(parent_dir):
    subdir_path = os.path.join(parent_dir, subdir)
    
    if os.path.isdir(subdir_path):  # Ensure it's a subdirectory
        for file in os.listdir(subdir_path):
            if file.startswith('Head_data_study') and file.endswith('.csv'):
                # Extract information from filename
                parts = file.split('-')
                study = parts[0].split('_')[2]
                participant = int(parts[1])
                date = parts[2]
                size = int(parts[3])
                pin_type = int(parts[4])
                repeat_time = int(parts[5].split('.')[0])

                # Generate PINEntry and Saccades filenames
                pin_entry_file = os.path.join(subdir_path, f"PINEntry_{study}-{participant}-{date}-{size}-{pin_type}-{repeat_time}.txt")
                saccades_file = os.path.join(subdir_path, f"Saccades_{study}-{participant}-{date}-{size}-{pin_type}-{repeat_time}.txt")
                
                # Read PINEntry and Saccades files
                try:
                    with open(pin_entry_file, 'r') as f:
                        pin_entry = f.read().strip().split('-')
                        pin_entry.pop(0)
                except FileNotFoundError:
                    print(f"File not found: {pin_entry_file}, skipping this entry.")
                    continue

                try:
                    with open(saccades_file, 'r') as f:
                        saccades_ranges = f.read().strip()
                except FileNotFoundError:
                    print(f"File not found: {saccades_file}, skipping this entry.")
                    continue

                # Read Head_data file
                head_data_file = os.path.join(subdir_path, file)
                head_data = pd.read_csv(head_data_file)

                # Get fixation time ranges
                fixation_ranges = get_fixation_ranges(saccades_ranges, len(head_data))

                # Calculate THM for each fixation range
                for fixation_range, pin in zip(fixation_ranges, pin_entry):
                    pin = int(pin)
                    target_distance = target_distance_mapping.get((size, pin), None)
                    if target_distance == 0:
                        continue
                    if target_distance is not None:
                        fixation_data = head_data.iloc[fixation_range[0]:fixation_range[1] + 1]
                        thms = calculate_thm(fixation_data)
                        normalized_thms = [thm / target_distance for thm in thms]  # Normalize each THM by target_distance
                        
                        if participant not in thm_data:
                            thm_data[participant] = []
                        thm_data[participant].extend(normalized_thms)

# Sort the participants by their number
sorted_thm_data = dict(sorted(thm_data.items()))

# Plot all participants' gain distributions in a single figure
plt.figure(figsize=(8, 6))
colors = cm.rainbow(np.linspace(0, 1, len(sorted_thm_data)))  # Color mapping for participants

# Plot each participant's gain distribution
for i, (participant, gains) in enumerate(sorted_thm_data.items()):
    # Filter out participants 17 and above if needed
    if participant >= 17:
        continue

    kde = gaussian_kde(gains, bw_method=0.2)
    x_grid = np.linspace(0, 2.5, 1000)

    # Plot each participant's gain distribution
    plt.plot(x_grid, kde(x_grid), color=colors[i], linestyle='-', label=f'P{participant}')

# Add labels, legend, and title
plt.xlabel('Gain (Head Movement Amplitude / Target Amplitude)', fontsize=16)
plt.ylabel('Probability Density', fontsize=16)
plt.xlim(0, 2.5)

# Set legend
plt.legend(loc='best', title='Participants', fontsize=16, title_fontsize=16, ncol=2, handletextpad=1)

# Add title below the plot
plt.figtext(0.5, -0.1, 'Gain Distribution for All Participants Across All Target Distances', ha='center', fontsize=36)

# Adjust layout to minimize borders
plt.tight_layout()

# Show plot
plt.show()

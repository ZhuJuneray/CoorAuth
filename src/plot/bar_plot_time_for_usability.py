import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from math import sqrt
plt.rcParams.update({'font.size': 20})
postures = ['Sitting', 'Standing']
# # login
# average_times = [3.863, 3.442]
# sd_times = [1.248, 1.089]
# se_times = [0.064382246, 0.055935798]
# # login

# registration
average_times = [22.43, 20.24]
sd_times = [5.584, 4.954]
se_times = [0.286, 0.2541]
# registration

# # error rate  
# average_times = [0.025, 0.01]
# sd_times = [0.157, 0.1]
# se_times = [0.011101576, 0.007071068]
# # error rate

# # error rate percentage
# average_times = [2.5, 1]
# sd_times = [15.7, 10]
# se_times = [1.1101576, 0.7071068]
# # error rate percentage

print(f"se_times: {se_times}")
# Plot the bar chart and save it
colors = plt.cm.Blues(np.linspace(0.15, 0.85, len(postures)+2))
width = 0.5
fig, ax = plt.subplots(figsize=(6, 5))
ax.bar(postures, average_times, width=width, yerr=se_times, capsize=40, color=colors, edgecolor='grey')
ax.set_ylabel('Registration Time(s)')
ax.set_xticklabels(postures)  # Ensure every size is marked
# Remove right and top spines/borders
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.grid(True, linestyle='--', alpha=0.7, color='grey')
ax.set_ylim(0, 28)
# ax.set_xlabel('(a)')
plt.tight_layout()
plt.show()
plt.close()

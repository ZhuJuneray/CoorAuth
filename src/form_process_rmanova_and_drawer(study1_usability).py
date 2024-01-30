import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

import csv, os, re, sys
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM
import pandas as pd
import pingouin as pg

# Define the path to your input CSV file and output CSV file
input_csv_file = os.path.join(os.getcwd(), 'data/VRAuth1/S1form.csv')

df = pd.read_csv(input_csv_file)

# Prepare to store results
rmanova_results = {}
posthoc_results = {}

question_list = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7']

# Performing Repeated Measures ANOVA for each question (Q1 to Q7)
for question in question_list:
    df_subset = df[['P', 'S', question]].copy()
    df_subset = df_subset.rename(columns={question: 'Score'})
    rmanova = AnovaRM(df_subset, 'Score', 'P', within=['S']).fit()
    rmanova_results[question] = rmanova.summary()

    posthoc = pg.pairwise_tests(data=df_subset, dv='Score', within='S', subject='P', padjust='bonferroni')
    posthoc_results[question] = posthoc



# Prepare an empty dictionary to store the results
results = {}
# Loop through each question (Q) and each S value (S)
for question in question_list:
    for s_value in [1,2,3,4]:
        subset = df[(df['S'] == s_value)][question]
        # Calculate the mean (M) and standard error (SE) for the filtered subset
        mean = subset.mean()
        std_error = subset.sem()

        # Store the results in the dictionary
        results[(question, s_value)] = {'M': mean, 'SE': std_error}

with open('form_process_rmanova.txt', 'w') as file:  # 将print内容保存到文件
    # 保存当前的标准输出
    original_stdout = sys.stdout
    # 将标准输出重定向到文件
    sys.stdout = file

    print(rmanova_results)
    print(posthoc_results)

    # Print the results
    for key, value in results.items():
        print(f'Question: {key[0]}, S: {key[1]}, M: {value["M"]}, SE: {value["SE"]}')

    sys.stdout = original_stdout



# Extracting the unique questions and S values
questions = sorted(set(key[0] for key in results))
s_values = sorted(set(key[1] for key in results))

# 创建一个颜色列表，用于存储渐进的蓝色
colors = plt.cm.Blues(np.linspace(0.15, 0.85, len(s_values)+2))

# Preparing data for plotting
mean_scores = {s: [results[(q, s)]['M'] for q in questions] for s in s_values}
se_scores = {s: [results[(q, s)]['SE'] for q in questions] for s in s_values}

# Plotting
fig, ax = plt.subplots(figsize=(15, 8))
width = 0.2  # the width of the bars
ind = np.arange(len(questions))  # the x locations for the groups

# Use the colors from the palette
for i, s in enumerate(s_values):
    ax.bar(ind - width/2. + (i-1)*width, mean_scores[s], width, label='S'+str(s), color=colors[i], edgecolor='grey')
    ax.errorbar(ind - width/2. + (i-1)*width, mean_scores[s], yerr=se_scores[s], fmt='none', ecolor='k', capsize=5, alpha=0.5)

# Add some text for labels, title, and custom x-axis tick labels, etc.
ax.set_ylabel('Mean 7-point Likert Scale Score', fontsize=16)
# ax.set_title('Scores by question and S value')
ax.set_xticks(ind)
questions_with_description = ['Q1: Mental Demand', 'Q2: Physical Demand', 'Q3: Performance', 'Q4: Frustration', 'Q5:Task Ease', 'Q6: Input Speed', 'Q7: Error Proneness',]
ax.set_xticklabels(questions, fontsize=16)
ax.set_xlabel('Questions', fontsize=16)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 16)

# for i in range(len(questions)): # draw all Q
for i in [1, 5]: # only draw those Q with Ridit significance
    significance_count = [0] * len(s_values) # 计数器，用于记录每个s值的significance line的个数
    height_for_draw_significance_line = [0] * len(s_values)
    for delta_i in range(1 , len(s_values)): # 为了让循环以跨度从小到大递增，即使得图片的significance line短的在下 长的在上
        for j in range(len(s_values)-1):
            k = j + delta_i
            if k < len(s_values):
                p_value = (posthoc_results[f"Q{i+1}"][(posthoc_results[f"Q{i+1}"]['A'] == j+1) & (posthoc_results[f"Q{i+1}"]['B'] == k+1)]['p-corr'].values)[0] # 从posthoc_results中取出p值
                if p_value < 0.05: 
                    # Draw a line
                    height_list = [mean_scores[s_values[temp_index]][i] + se_scores[s_values[temp_index]][i] for temp_index in range(len(s_values))] # 每个s值的高度
                    highest_index = height_list.index(max([height_list[temp_index] for temp_index in range(j, k+1)])) # 跨度内最高点的index
                    if significance_count[highest_index] == 0: # 跨度内最高点未被占用
                        height_for_draw_significance_line[highest_index] = height_list[highest_index]+0.1 # 0.1是为了让significance line不与errorbar重合
                    else:
                        height_for_draw_significance_line[highest_index] += 0.15 # 跨度内最高点被占用，往上移动， 0.15是随便定的
                    significance_count[highest_index] += 1 # 跨度内最高点被占用，计数器+1
                    ax.plot([ind[i] - width/2. + (j-1)*width, ind[i] - width/2. + (k-1)*width],
                            [height_for_draw_significance_line[highest_index], height_for_draw_significance_line[highest_index]], color= 'grey', linewidth=1)
                    # 在significance line两端添加短竖线
                    dot_height = 0.03
                    ax.plot([ind[i] - width/2. + (j-1)*width, ind[i] - width/2. + (j-1)*width],
                            [height_for_draw_significance_line[highest_index] - dot_height, height_for_draw_significance_line[highest_index] + dot_height], color='grey', linewidth=1)
                    ax.plot([ind[i] - width/2. + (k-1)*width, ind[i] - width/2. + (k-1)*width],
                            [height_for_draw_significance_line[highest_index] - dot_height, height_for_draw_significance_line[highest_index] + dot_height], color='grey', linewidth=1)
                    # Add asterisks for significance
                    ax.text((2*ind[i] - width + (j+k-2)*width) / 2., height_for_draw_significance_line[highest_index],
                            '***' if p_value < 0.001 else ('**' if p_value < 0.01 else '*'), ha='center')

# Draw horizontal grey dashed lines at each integer height
y_max = int(max(score['M'] + score['SE'] for score in results.values())) + 1
for y in range(y_max):
    ax.axhline(y=y, color='grey', linewidth=0.8, linestyle='--')


# Define and create the plot folder
plot_folder = os.path.join("result/", "study1")
if not os.path.exists(plot_folder):
    os.makedirs(plot_folder)

# Define the plot filename
plot_filename = f"form.png"
fig.savefig(os.path.join(plot_folder, plot_filename))
plt.close(fig)
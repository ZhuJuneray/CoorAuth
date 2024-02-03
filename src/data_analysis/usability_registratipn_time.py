import pandas as pd
import numpy as np

repeat_times_for_group = 100

# 对每个组重复n次随机选择和计算，每次计算加上四个在0.5到1之间的随机数
def repeat_and_adjust(x):
    sums = []
    for _ in range(repeat_times_for_group):
        # 随机选择5个值，如果组内少于5个值，则选择所有值
        sample_sum = x.sample(n=min(5, len(x))).sum()
        # 加上四个在0.5到1之间的随机数的和
        adjustment = np.random.uniform(0.5, 1, 4).sum()
        sums.append(sample_sum + adjustment)
    return np.mean(sums)

def repeat_and_adjus_origin(x):
    sums = []
    for _ in range(repeat_times_for_group):
        # 随机选择5个值，如果组内少于5个值，则选择所有值
        sample_sum = x.sample(n=min(5, len(x))).sum()
        # 加上四个在0.5到1之间的随机数的和
        adjustment = np.random.uniform(0.5, 1, 4).sum()
        sums.append(sample_sum + adjustment)
    return sums

# 假设的数据结构，按照用户提供的结构创建DataFrame
data = pd.read_excel('user_size_pin_time_study1_sitting.xlsx')
df = pd.DataFrame(data)

# 分组处理
grouped = df.groupby(['User', 'Pin'])
grouped2 = df.groupby(['User', 'Pin'])

# 应用上述函数到每个组
adjusted_sums = grouped['Time'].apply(repeat_and_adjust)
adjusted_sums_origin = grouped2['Time'].apply(repeat_and_adjus_origin)
# print(f"adjusted_sums_origin: {adjusted_sums_origin}")
all_values_flat = adjusted_sums_origin.values.flatten()
all_values = [element for row in [i for i in adjusted_sums_origin.values] for element in row]
# save all_values to excel
df_all_values = pd.DataFrame(all_values)
df_all_values.to_excel("sitting_registration.xlsx", index=False)
# print(f"all_values_flat: {all_values}")


# 计算所有调整后的和的平均值
# print(f"adjusted_sums: {adjusted_sums}")
average_adjusted_sum = np.array(all_values).mean()
sd_adjusted_sum = np.array(all_values).std()
se_adjusted_sum = sd_adjusted_sum/np.sqrt(len(all_values))
print(f"repeat times for each (user, pin) pair: {repeat_times_for_group}")
print(f"average registration time: {average_adjusted_sum}")
print(f"sd registration time: {sd_adjusted_sum}")
print(f"se registration time: {se_adjusted_sum}")
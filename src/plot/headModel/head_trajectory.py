import csv
import matplotlib.pyplot as plt
import os
import matplotlib.patches as patches
import matplotlib.colors as mcolors

def desaturate_color(color, proportion):
    """降低颜色的饱和度"""
    # 将颜色转换为RGB格式
    rgb = mcolors.to_rgb(color)
    # 计算灰度（平均值）
    grey = sum(rgb) / 3
    # 混合原颜色和灰度
    new_rgb = [grey + proportion * (c - grey) for c in rgb]
    return new_rgb

# 四个角的坐标
corners = [(1, 1), (-1, 1), (-1, -1), (1, -1)]

# 3x3网格的九个点的相对坐标
grid_points = [(-1, 1), (0, 1), (1, 1),
               (-1, 0), (0, 0), (1, 0),
               (-1, -1), (0, -1), (1, -1)]

# 文件夹路径
folder_path = os.path.join(os.getcwd(), "src", "plot", "headModel", "P100")
title = '(a)'

# 获取文件夹中所有CSV文件
csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

# 创建图形和坐标轴
fig, ax = plt.subplots(figsize=(5,5))

# 为底部标题调整子图布局
fig.subplots_adjust(bottom=0.2)
ax.set_aspect('equal')

# 自定义蓝色和紫色的颜色循环
# original_colors = ['plum', 'purple', 'mediumpurple', 'darkblue', 'plum']
original_colors = ['red', 'lime', 'royalblue', 'yellow', 'purple', 'orange', 'pink', 'brown', 'grey', 'black']
# 降低颜色饱和度
color_cycle = [desaturate_color(color, 1) for color in original_colors]

# 绘制3x3网格的点
for point in grid_points:
    # ax.scatter(point[0], point[1], color='blue')
    circle = patches.Circle((point[0], point[1]), 0.2, color='grey', ec='black', lw=1)
    ax.add_patch(circle)
    # 添加阴影效果
    shadow = patches.Circle((point[0]-0.05, point[1]-0.05), 0.2, color='grey', alpha=0.3)
    ax.add_patch(shadow)

# 遍历每个文件
for i, csv_file in enumerate(csv_files):
    file_path = os.path.join(folder_path, csv_file)

    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过表头
        data = [list(map(float, row)) for row in reader]

    # 提取 x 和 y 坐标
    x = [row[0] for row in data]
    y = [row[1] for row in data]

    # 绘制散点图，为每个文件使用不同的颜色
    ax.scatter(x, y, color=color_cycle[i % len(color_cycle)], alpha = 0.5, label=csv_file)




# 绘制四个角的点
# for corner in corners:
#     plt.scatter(corner[0], corner[1], color='red')



# 连接四个角的点，形成一个方框
# corners.append(corners[0])  # 闭合路径
# corner_x, corner_y = zip(*corners)

# 添加标题和坐标轴标签
# ax.set_title('Points and 3x3 Grid')
# ax.set_xlabel('X Coordinate')
# ax.set_ylabel('Y Coordinate')

# 显示图例
# ax.legend()
# plt.tight_layout()

ax.axis('off')  # 不显示坐标轴
# fig.text(0.5, 0.05, title, ha='center', fontsize=26)

# 显示图形
plt.show()

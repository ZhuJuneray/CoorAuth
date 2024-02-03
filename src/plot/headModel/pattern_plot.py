import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines  # 导入matplotlib.lines模块
from matplotlib.patches import FancyArrowPatch

def draw_pattern(patterns, title=None, colors=['r']):
    # 创建图形和坐标轴
    fig, ax = plt.subplots(figsize=(5, 5))

    # 为底部标题调整子图布局
    fig.subplots_adjust(bottom=0.2)

    # 绘制带阴影的灰色小球
    for x in range(3):
        for y in range(3):
            # 创建灰色的小球
            circle = patches.Circle((x, y), 0.2, color='grey', ec='black', lw=1)
            ax.add_patch(circle)
            # 添加阴影效果
            shadow = patches.Circle((x - 0.05, y - 0.05), 0.2, color='grey', alpha=0.3)
            ax.add_patch(shadow)
            ax.text(x, y, str(10 - (3 * y + abs(2 - x)  + 1)), color='white', fontsize=20, ha='center', va='center', alpha=0.7)
    
    multi_shift = [0.04, 0.04]
    line_width = 5

    for pattern in patterns:
        pattern_num = patterns.index(pattern)
        # 绘制连接点的线（如果需要）
        for i in range(len(pattern) - 1):
            point_start = ((int(pattern[i])-1)%3+multi_shift[0]*pattern_num, 2-(int(pattern[i])-1)//3+multi_shift[1]*pattern_num)
            point_end = ((int(pattern[i + 1])-1)%3+multi_shift[0]*pattern_num, 2-(int(pattern[i + 1])-1)//3+multi_shift[1]*pattern_num)

            # 检查是否是最后一段线
            if i == len(pattern) - 2:
                # 仅在最后一段线上添加箭头
                arrow = FancyArrowPatch(point_start, (point_end[0], point_end[1]-multi_shift[1]*pattern_num), color=colors[pattern_num], arrowstyle='->', mutation_scale=30, linewidth=line_width, alpha=0.9)
                ax.add_patch(arrow)
            else:
                # 绘制不带箭头的线
                line = mlines.Line2D([point_start[0], point_end[0]], [point_start[1], point_end[1]], color=colors[pattern_num], lw=line_width, alpha=0.9)
                ax.add_line(line)

    # else:
    #     # 绘制连接点的线（如果需要）
    #     for i in range(len(pattern) - 1):
    #         point_start = ((int(pattern[i])-1)%3, 2-(int(pattern[i])-1)//3)
    #         point_end = ((int(pattern[i + 1])-1)%3, 2-(int(pattern[i + 1])-1)//3)

    #                 # 检查是否是最后一段线
    #         if i == len(pattern) - 2:
    #             # 仅在最后一段线上添加箭头
    #             arrow = FancyArrowPatch(point_start, point_end, color='red', arrowstyle='->', mutation_scale=30, linewidth=2)
    #             ax.add_patch(arrow)
    #         else:
    #             # 绘制不带箭头的线
    #             # line = patches.FancyArrowPatch(point_start, point_end, color='red', arrowstyle='-', mutation_scale=30, linewidth=2)
    #             # ax.add_patch(line)
    #             line = mlines.Line2D([point_start[0], point_end[0]], [point_start[1], point_end[1]], color='red', lw=2)
    #             ax.add_line(line)



        # arrow = FancyArrowPatch(point_start, point_end, color='red', arrowstyle='->', mutation_scale=40)
        # ax.add_patch(arrow)
        # line = mlines.Line2D([point_start[0], point_end[0]], [point_start[1], point_end[1]], color='red', lw=2)
        # ax.add_line(line)

    # 设置图形属性
    ax.set_aspect('equal', adjustable='box')
    plt.xlim(-1, 3)
    plt.ylim(-1, 3)
    ax.axis('off')  # 不显示坐标轴
    # 添加底部标题
    if title:
        fig.text(0.5, 0.05, title, ha='center', fontsize=26)
    fig.tight_layout()
    plt.show()

# 定义一个解锁模式，例如：从(0, 0)连接到(1, 1)，再到(2, 0)
pattern = '12369'
patterns = ['1235', '12369', '1236', ] # ['1236', '1235', '12369']
color_list = ['lime', 'orangered', 'royalblue', 'yellow', 'purple', 'orange', 'pink', 'brown', 'grey', 'black']
# draw_pattern(pattern, f'Pattern3 = "{pattern}"')
draw_pattern(patterns=patterns, title=f'', colors=color_list)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 创建一个示例的四元组坐标数据（替换为你自己的数据）
# 这里假设有100个四元组数据点
num_points = 100
x = [0.1 * i for i in range(num_points)]  # 替换为你的x坐标数据
y = [0.2 * i for i in range(num_points)]  # 替换为你的y坐标数据
z = [0.3 * i for i in range(num_points)]  # 替换为你的z坐标数据
w = [0.4 * i for i in range(num_points)]  # 替换为你的w坐标数据

# 创建一个三维坐标图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制四元组坐标数据点
ax.scatter(x, y, z, c=w, cmap='viridis')

# 添加坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 显示图形
plt.show()

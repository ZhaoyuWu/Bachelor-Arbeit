import gym
import numpy as np
from ENV import PathEnvironment
import matplotlib.pyplot as plt


def plot_observation(ax, accumulated_obs):
    # 由于 accumulated_obs 现在是一个 numpy 数组，我们可以直接用切片来获取 x 和 y
    x = accumulated_obs[:, 0]
    y = accumulated_obs[:, 1]
    center_x = accumulated_obs[:, 2]
    center_y = accumulated_obs[:, 3]

    # 绘制笔迹点并按顺序将它们连接起来
    # ax.plot(x, y, 'r-o', label='Stroke Points')  # 使用红色线和圆圈表示笔迹点
    # 绘制路径中心点并按顺序将它们连接起来
    # ax.plot(center_x, center_y, 'b-x', label='Path Center')  # 使用蓝色线和叉号表示路径中心

    # 使用 scatter 方法绘制笔迹点，不将它们连接起来
    ax.scatter(x, y, color='red', marker='o', label='Stroke Points')  # 使用红色圆圈表示笔迹点

    # 使用 scatter 方法绘制路径中心点，不将它们连接起来
    ax.scatter(center_x, center_y, color='blue', marker='x', label='Path Center')  # 使用蓝色叉号表示路径中心

    # 特别标记初始点
    ax.scatter(x[0], y[0], s=100, c='yellow', edgecolors='black', zorder=5, label='Start Point')
    # ax.scatter(center_x[0], center_y[0], s=100, c='green', edgecolors='black', zorder=5, label='Start Center')

    # 设置标题和标签
    ax.set_title('Observation')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()

    # 显示网格
    ax.grid(True)


# 导入你注册的环境
env = PathEnvironment()

# 重置环境
obs = env.reset()

# 初始化一个数组来累积所有的观测点
accumulated_obs = obs.copy()

# 打印初始状态
print("Init state：")
print(obs)
# 初始化一个列表来收集每一步的奖励
rewards = []

# 迭代执行一些步骤并观察结果
for i in range(40):
    action = np.array([-0.1])  # 定义一个合法的动作，范围在[-1, 1]之间
    obs, reward, done, info = env.step(action)  # 执行动作
    accumulated_obs = np.vstack((accumulated_obs, obs))
    # 收集奖励
    rewards.append(reward)

# 创建图形和子图用于绘制观测点
fig, axs = plt.subplots(2, 1, figsize=(10, 10))  # 创建两个子图

# 绘制包含所有状态的观测点图形
plot_observation(axs[0], accumulated_obs)

# 绘制奖励变化曲线
axs[1].plot(rewards, 'g^-', label='Reward')  # 使用绿色线和三角形标记表示奖励
axs[1].set_title('Reward Over Time')
axs[1].set_xlabel('Step')
axs[1].set_ylabel('Reward')
axs[1].legend()
axs[1].grid(True)

# 显示图像
plt.tight_layout()  # 调整子图间距
# 迭代执行一些步骤并观察结果
for i in range(40):
    action = np.array([-0.1])  # 定义一个合法的动作，范围在[-1, 1]之间
    obs, reward, done, info = env.step(action)  # 执行动作

    # 将新的观测点累积到之前的观测点数组中
    accumulated_obs = np.vstack((accumulated_obs, obs))

    print(f"Episode {i + 1}：")
    print("State：", obs)
    print("Reward：", reward)
    print("Terminate：", done)
    print("info：", info)
    print()

# 显示图像
plt.show()

# 关闭环境
env.close()

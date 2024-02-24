import gym
import numpy as np
from ENV import PathEnvironment
import matplotlib.pyplot as plt


def plot_observation(ax, accumulated_obs):
    # stroke coordinate
    x = accumulated_obs[:, 0]
    y = accumulated_obs[:, 1]
    # path coordinate
    center_x = accumulated_obs[:, 2]
    center_y = accumulated_obs[:, 3]

    # draw stroke
    ax.scatter(x, y, color='red', marker='o', label='Stroke Points')  # 使用红色圆圈表示笔迹点

    # draw path
    ax.scatter(center_x, center_y, color='blue', marker='x', label='Path Center')  # 使用蓝色叉号表示路径中心

    # Start point
    ax.scatter(x[0], y[0], s=100, c='yellow', edgecolors='black', zorder=5, label='Start Point')
    # ax.scatter(center_x[0], center_y[0], s=100, c='green', edgecolors='black', zorder=5, label='Start Center')

    ax.set_title('Observation')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()

    ax.grid(True)


# import environment
env = PathEnvironment()

# reset environment
obs = env.reset()

accumulated_obs = obs.copy()

print("Init state：")
print(obs)
rewards = []

for i in range(40):
    action = np.array([-0.1])  # Action each loop
    obs, reward, done, info = env.step(action)
    accumulated_obs = np.vstack((accumulated_obs, obs))
    # rewards
    rewards.append(reward)

    print(f"Episode {i + 1}：")
    print("State：", obs)
    print("Reward：", reward)
    print("Terminate：", done)
    print("info：", info)
    print()

fig, axs = plt.subplots(2, 1, figsize=(10, 10))
plot_observation(axs[0], accumulated_obs)

# plot rewards
axs[1].plot(rewards, 'g^-', label='Reward')
axs[1].set_title('Reward Over Time')
axs[1].set_xlabel('Step')
axs[1].set_ylabel('Reward')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()

plt.show()

env.close()

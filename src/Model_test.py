import matplotlib.pyplot as plt
from ENV import PathEnvironment
from DDPG import DDPG

a_dim = 1  
s_dim = 15
a_bound = 1

ddpg = DDPG(a_dim, s_dim, a_bound)

plt.show(block=True)

env = PathEnvironment()
ddpg.load_ckpt()

plt.ion()
fig, ax = plt.subplots(figsize=(10, 6))

actionset = (-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2)

for j in range(len(actionset)):
    s = env.reset()
    actions = []
    widths = []
    rewards = []
    steps = []

    for i in range(100):
        action = ddpg.choose_action(s)
        actions.append(actionset[j])
        widths.append(env.avg_width)
        steps.append(i)

        s, r, done, info = env.step(action)
        rewards.append(r)

    # 绘制每个动作对应的奖励曲线
    ax.plot(rewards, linestyle='--', label=f'Action: {actionset[j]}')

s = env.reset()
rewards_action0 = []

for i in range(100):
    action = ddpg.choose_action(s)
    actions.append(action[0])  # apply DDPG
    widths.append(env.avg_width)
    steps.append(i)

    s, r, done, info = env.step(action)
    rewards_action0.append(r)

ax.plot(rewards_action0, label='Action: DDPG')

ax.set_xlabel('Step')
ax.set_ylabel('Reward')
ax.set_title('Reward Changes under Different Actions')
ax.legend()

plt.pause(0.1)

plt.ioff()
plt.show()

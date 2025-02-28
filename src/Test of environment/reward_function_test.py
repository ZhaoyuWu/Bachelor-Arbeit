from ENV import PathEnvironment
import matplotlib.pyplot as plt

a_dim = 1
s_dim = 15
a_bound = 1

env = PathEnvironment()
# 1
env.set_parameters(0.6813977389516958,
                   0.39874666682693943,
                   0.02237240909822033,
                   0.0672254672097416,
                   -0.012994991611979543,
                   -0.009778060955385063,
                   0.07022049694376897)

# 2
# env.set_parameters(0.23747276688453156,
#                        0.1386253751166414,
#                        0.031218818394109227,
#                        0.029027581576196587,
#                        -0.003849595086849577,
#                        -0.00699335289743980,
#                        0.0238051992111944)

# 3
# env.set_parameters(0.5233463035019454,
#                        0.47640023722986047,
#                        0.013679382626220541,
#                        0.00729660258990886,
#                        -0.008056598615398376,
#                        -0.007745287704928358,
#                        0.0065203282945992865)

actionset = (-1, -0.8, -0.6, -0.4, -0.2, 0)

batch_length = 20

iterate = 100

average_rewards = []

init_widths = [ 1.5, 2, 2.5, 3, 3.5]

# Store results
results = {}

# Loop through different initial widths
for width in init_widths:
    print(f"Testing with init_width = {width}")
    # Initialize total rewards
    total_rewards_ddpg = 0
    average_rewards = []

    # Loop through each action in action set
    for action in actionset:
        total_reward = 0
        for _ in range(iterate):
            s = env.reset()
            for i in range(batch_length):
                s, r, done, info = env.step(action)
                env.set_width(width)
                total_reward += r
                if done:
                    break
        average_reward = total_reward / iterate
        average_rewards.append(average_reward)
        print(f'Average reward for action {action}: {average_reward}')

    # Store results for this width
    results[width] = average_rewards

# Print or save results for analysis
print("Results:")
for width, rewards in results.items():
    print(f"Init_width = {width}: {rewards}")

for width, rewards in results.items():
    actions_with_ddpg = list(actionset)
    plt.plot(rewards[:-1], label=f'Init_width={width}')
    #actions_with_ddpg,
    #plt.axhline(y=rewards[-1], color='r', linestyle='--')

plt.xlabel('Actions')
plt.ylabel('Average Reward')
plt.title('Average Reward vs Actions for Different Initial Widths')
plt.legend()
plt.show()
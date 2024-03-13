from ENV import PathEnvironment
from DDPG import DDPG
import numpy as np
import tensorflow as tf

env = PathEnvironment()
env = env.unwrapped

env.set_parameters(0.8, 0.6, 0.3, 0.3, -0.2, -0.3, 0.1)

MAX_EPISODES = 100  # total number of episodes for training
MAX_EP_STEPS = 50  # total number of steps for each episode
TEST_PER_EPISODES = 10  # test the model per episodes
MEMORY_CAPACITY = 10000  # size of replay buffer

def train_or_test_ddpg(env, ddpg, args):
    """
    Train or test DDPG agent.

    Args:
        env: The environment instance.
        ddpg: The DDPG agent instance.
        args: The command line arguments.

    Returns:
        None
    """
    state_shape = env.observation_space.shape
    s_dim = np.prod(state_shape)
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high

    ddpg = DDPG(a_dim, s_dim, a_bound)

    # train
    if args.train:
        for i in range(MAX_EPISODES):
            s = env.reset()
            for j in range(MAX_EP_STEPS):
                a = ddpg.choose_action(s)
                s_, r, done, info = env.step(a)
                ddpg.store_transition(s, a, r / 10, s_)
                if ddpg.pointer > MEMORY_CAPACITY:
                    ddpg.learn()
                if done:
                    break
                s = s_

    # test
    else:
        s = env.reset()
        for i in range(200):
            action = ddpg.choose_action(s)
            s, r, done, info = env.step(action)
            if done:
                break

    if __name__ == '__main__':
        env = PathEnvironment()
        env.set_parameters(0.8, 0.6, 0.3, 0.3, -0.2, -0.3, 0.1)
        env = env.unwrapped
        ddpg = DDPG(a_dim, s_dim, a_bound)

        train_or_test_ddpg(env, ddpg, args)


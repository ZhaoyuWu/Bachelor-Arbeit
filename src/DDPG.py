"""
Deep Deterministic Policy Gradient (DDPG)
-----------------------------------------
An algorithm concurrently learns a Q-function and a policy.
It uses off-policy data and the Bellman equation to learn the Q-function,
and uses the Q-function to learn the policy.

Reference
---------
Deterministic Policy Gradient Algorithms, Silver et al. 2014
Continuous Control With Deep Reinforcement Learning, Lillicrap et al. 2016
MorvanZhou's tutorial page: https://morvanzhou.github.io/tutorials/

Environment
-----------
Openai Gym Pendulum-v0, continual action space

Prerequisites
-------------
tensorflow >=2.0.0a0
tensorflow-probability 0.6.0
tensorlayer >=2.0.0

To run
------
python tutorial_DDPG.py --train/test

"""

import argparse
import os
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from ENV import PathEnvironment

import tensorlayer as tl

parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='test', action='store_false')
args = parser.parse_args()

#####################  hyper parameters  ####################

ENV_NAME = 'Pendulum-v1'  # environment name
RANDOMSEED = 1  # random seed

LR_A = 0.001  # learning rate for actor
LR_C = 0.002  # learning rate for critic
GAMMA = 0.9  # reward discount
TAU = 0.01  # soft replacement
MEMORY_CAPACITY = 10000  # size of replay buffer
BATCH_SIZE = 32  # update batchsize

MAX_EPISODES = 50  # total number of episodes for training
MAX_EP_STEPS = 20  # total number of steps for each episode
TEST_PER_EPISODES = 10  # test the model per episodes
VAR = 1  # control exploration


###############################  DDPG  ####################################

class DDPG(object):
    """
    DDPG class
    """

    def __init__(self, a_dim, s_dim, a_bound):
        # memory saves the data：
        # MEMORY_CAPACITY，s_dim * 2 + a_dim + 1：two states，one action，one reward
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound

        W_init = tf.random_normal_initializer(mean=0, stddev=0.3)
        b_init = tf.constant_initializer(0.1)

        # Actor network，input s，output a
        def get_actor(input_state_shape, name=''):
            """
            Build actor network
            :param input_state_shape: state shape, e.g., [None, 3, 5]
            :param name: name
            :return: act
            """
            inputs = tl.layers.Input(input_state_shape, name='A_input')
            x = tl.layers.Dense(n_units=15, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='A_l1')(inputs)
            x = tl.layers.Dense(n_units=a_dim, act=tf.nn.tanh, W_init=W_init, b_init=b_init, name='A_a')(x)
            x = tl.layers.Lambda(lambda x: np.array(a_bound) * x)(x)
            return tl.models.Model(inputs=inputs, outputs=x, name='Actor' + name)

        # Critic network，input s，a。output Q(s,a)
        def get_critic(input_state_shape, input_action_shape, name=''):
            """
            Build critic network
            :param input_state_shape: state
            :param input_action_shape: act
            :param name: name
            :return: Q value Q(s,a)
            """
            s = tl.layers.Input(input_state_shape, name='C_s_input')
            a = tl.layers.Input(input_action_shape, name='C_a_input')
            x = tl.layers.Concat(1)([s, a])
            x = tl.layers.Dense(n_units=30, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='C_l1')(x)
            x = tl.layers.Dense(n_units=1, W_init=W_init, b_init=b_init, name='C_out')(x)
            # print("Q: ",x)
            return tl.models.Model(inputs=[s, a], outputs=x, name='Critic' + name)

        self.actor = get_actor([None, s_dim])
        self.critic = get_critic([None, s_dim], [None, a_dim])
        self.actor.train()
        self.critic.train()

        def copy_para(from_model, to_model):
            """
            Copy parameters for soft updating
            :param from_model: latest model
            :param to_model: target model
            :return: None
            """
            for i, j in zip(from_model.trainable_weights, to_model.trainable_weights):
                j.assign(i)

        # actor_target for later update
        self.actor_target = get_actor([None, s_dim], name='_target')
        copy_para(self.actor, self.actor_target)
        self.actor_target.eval()

        # critic_target
        self.critic_target = get_critic([None, s_dim], [None, a_dim], name='_target')
        copy_para(self.critic, self.critic_target)
        self.critic_target.eval()

        self.R = tl.layers.Input([None, 1], tf.float32, 'r')

        # ExponentialMovingAverage
        self.ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)  # soft replacement

        self.actor_opt = tf.optimizers.Adam(LR_A)
        self.critic_opt = tf.optimizers.Adam(LR_C)

    def ema_update(self):
        """
        ExponentialMovingAverage update
        """
        paras = self.actor.trainable_weights + self.critic.trainable_weights
        self.ema.apply(paras)
        for i, j in zip(self.actor_target.trainable_weights + self.critic_target.trainable_weights, paras):
            i.assign(self.ema.average(j))

    def choose_action(self, s):
        """
        Choose action
        :param s: state, assumed to be a numpy array of shape (3, 5).
        :return: action
        """
        # Correctly flatten the state s to a two-dimensional array of shape [1, 15]
        s_flattened = s.flatten().reshape(1, -1)

        action = self.actor(s_flattened.astype(np.float32))[0]
        return action

    def learn(self):
        """
        Update parameters
        :return: None
        """
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)  # Experiment replay
        bt = self.memory[indices, :]  # choose random experiment
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim:self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1:-self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        # Critic：

        # br + GAMMA * q_
        with tf.GradientTape() as tape:
            a_ = self.actor_target(bs_)
            q_ = self.critic_target([bs_, a_])
            y = br + GAMMA * q_
            q = self.critic([bs, ba])
            # print("Q: ",q)
            td_error = tf.losses.mean_squared_error(y, q)
        c_grads = tape.gradient(td_error, self.critic.trainable_weights)
        self.critic_opt.apply_gradients(zip(c_grads, self.critic.trainable_weights))

        # Actor：
        # Actor choose the action witch returns largest Q value
        with tf.GradientTape() as tape:
            a = self.actor(bs)
            q = self.critic([bs, a])
            a_loss = -tf.reduce_mean(q)  # gradient increase
        a_grads = tape.gradient(a_loss, self.actor.trainable_weights)
        self.actor_opt.apply_gradients(zip(a_grads, self.actor.trainable_weights))

        self.ema_update()

    # 保存s，a，r，s_
    def store_transition(self, s, a, r, s_):
        """
        Store data in data buffer
        :param s: state
        :param a: act
        :param r: reward
        :param s_: next state
        :return: None
        """

        # s = s.astype(np.float32)
        # s_ = s_.astype(np.float32)
        s = s.flatten()
        s_ = s_.flatten()
        transition = np.hstack((s, a, [r], s_))

        # The pointer is a record of how much data has come in.。
        # index is the location of the latest incoming data.。
        # It's a loop, and when MEMORY_CAPACITY is full, index starts over at the bottom again
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        # transition
        self.memory[index, :] = transition
        self.pointer += 1

    def save_ckpt(self):
        """
        save trained weights
        :return: None
        """
        if not os.path.exists('model'):
            os.makedirs('model')

        tl.files.save_weights_to_hdf5('model/ddpg_actor.hdf5', self.actor)
        tl.files.save_weights_to_hdf5('model/ddpg_actor_target.hdf5', self.actor_target)
        tl.files.save_weights_to_hdf5('model/ddpg_critic.hdf5', self.critic)
        tl.files.save_weights_to_hdf5('model/ddpg_critic_target.hdf5', self.critic_target)

    def load_ckpt(self):
        """
        load trained weights
        :return: None
        """
        tl.files.load_hdf5_to_weights_in_order('model/ddpg_actor.hdf5', self.actor)
        tl.files.load_hdf5_to_weights_in_order('model/ddpg_actor_target.hdf5', self.actor_target)
        tl.files.load_hdf5_to_weights_in_order('model/ddpg_critic.hdf5', self.critic)
        tl.files.load_hdf5_to_weights_in_order('model/ddpg_critic_target.hdf5', self.critic_target)


if __name__ == '__main__':

    # env = gym.make(ENV_NAME)
    env = PathEnvironment()
    env = env.unwrapped

    # reproducible
    # env.seed(RANDOMSEED)
    np.random.seed(RANDOMSEED)
    tf.random.set_seed(RANDOMSEED)

    # Define state space, action space, range of action magnitude
    # s_dim = env.observation_space.shape[0]
    state_shape = env.observation_space.shape
    s_dim = np.prod(state_shape)
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high

    print('s_dim', s_dim)
    print('a_dim', a_dim)

    # ddpg
    ddpg = DDPG(a_dim, s_dim, a_bound)

    # train：
    if args.train:  # train

        reward_buffer = []  # reward each Episode
        t0 = time.time()  # time

        # 初始化图像显示
        plt.ion()
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))

        reward_history = []
        width_history = []


        for i in range(MAX_EPISODES):
            t1 = time.time()
            s = env.reset()
            ep_reward = 0

            for j in range(MAX_EP_STEPS):
                # Add exploration noise
                a = ddpg.choose_action(s)
                # print("DDPG Action: ", a)
                # To be able to keep the development going, here's another way to add exploration.
                # So need to need to create a normal distribution with a as the mean and VAR as the standard deviation, and then sample a from the normal distribution
                # Because a is the mean, the probability of a is maximized. But how big a is relative to the other probabilities by is adjusted by the VAR. Here we can actually add update the VAR to dynamically adjust the certainty of a
                # And then do the trimming
                # Or any suitable decay factor
                # print("DDGP Action",a)
                a = np.clip(np.random.normal(a, VAR), -2, 1)

                # print(env.step(a))
                s_, r, done, info = env.step(a)

                # print("state: ",s, "\naction: ",a,"\nrewards: ",r,"\nlast state: ",s_)
                # print("Action: ",a, "Reward: ", r)
                ddpg.store_transition(s, a, r / 10, s_)

                N = min(10, s.shape[0])
                width_history.append(np.mean(s[-N:, 4]))  # 假设环境中有一个属性 avg_width 记录当前的宽度

                # data full, start learning
                if ddpg.pointer > MEMORY_CAPACITY:
                    ddpg.learn()

                stroke_points = s_[:, :2]
                path_centers = s_[:, 2:4]

                axs[0].plot(stroke_points[:, 0], stroke_points[:, 1],'r-')
                axs[0].plot(path_centers[:, 0], path_centers[:, 1],'b-')

                plt.pause(0.1)

                if done:
                    break

                s = s_
                ep_reward += r  # total reward
                if j == MAX_EP_STEPS - 1:
                    print(
                        '\rEpisode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                            i, MAX_EPISODES, ep_reward,
                            time.time() - t1
                        ), end=''
                    )

                plt.show()

            reward_history.append(ep_reward)

            axs[1].clear()
            axs[1].plot(reward_history, label='Reward')
            axs[1].set_xlabel('Episode')
            axs[1].set_ylabel('Cumulative Reward')
            axs[1].legend()

            axs[2].clear()
            axs[2].plot(width_history, label='Width')
            axs[2].set_xlabel('Step')
            axs[2].set_ylabel('Width')
            axs[2].legend()

            plt.pause(0.01)

            # test
            if i and not i % TEST_PER_EPISODES:
                t1 = time.time()
                s = env.reset()
                ep_reward = 0
                for j in range(MAX_EP_STEPS):

                    a = ddpg.choose_action(s)  #When testing, we won't need to use a normal distribution, just a straight a will do.
                    s_, r, done, info = env.step(a)

                    s = s_
                    ep_reward += r
                    if j == MAX_EP_STEPS - 1:
                        print(
                            '\rEpisode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                                i, MAX_EPISODES, ep_reward,
                                time.time() - t1
                            )
                        )

                        reward_buffer.append(ep_reward)

            # print("EPR: ",reward_buffer)

        print('\nRunning time: ', time.time() - t0)
        ddpg.save_ckpt()
    plt.show(block=True)

    env = PathEnvironment()
    ddpg.load_ckpt()

    plt.ion()  # interactive mode
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    actions = []
    widths = []
    steps = []

    s = env.reset()

    for i in range(200):
        action = ddpg.choose_action(s)
        actions.append(action[0])
        widths.append(env.avg_width)
        steps.append(i)

        s, r, done, info = env.step(action)

        axs[0].clear()
        axs[0].plot(steps, actions, label='Action')
        axs[0].set_xlabel('Step')
        axs[0].set_ylabel('Action')
        axs[0].set_title('Action Changes')
        axs[0].legend()

        axs[1].clear()
        axs[1].plot(steps, widths, label='Width', color='red')
        axs[1].set_xlabel('Step')
        axs[1].set_ylabel('Width')
        axs[1].set_title('Width Changes')
        axs[1].legend()

        plt.pause(0.1)

        if done:
            break

    plt.ioff()
    plt.show()
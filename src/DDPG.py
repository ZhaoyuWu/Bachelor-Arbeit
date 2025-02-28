
import argparse
import os
import time
import sqlite3

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

#  hyper parameters  ======================================================

LR_A =  0.0001 # learning rate for actor
LR_C =  0.0005  # learning rate for critic
GAMMA = 0.9  # reward discount
TAU = 0.1  # soft replacement
MEMORY_CAPACITY = 5000  # size of replay buffer
BATCH_SIZE = 30 # update batchsize

MAX_EPISODES = 70  # total number of episodes for training
MAX_EP_STEPS = 50  # total number of steps for each episode
TEST_PER_EPISODES = 10  # test the model per episodes
VAR = 0.5  # control exploration


#  DDPG  ==================================================================

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

        self.loss_history = []

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
            x = tl.layers.Lambda(lambda x: x * (0.5 - (-1)) / (1 - (-1)) + (0.5 + (-1)) / 2)(x) # project action to [-1,0.5]
            #x = tl.layers.Lambda(lambda x: tf.round(x * 100) / 100.0)(x) # keep the action to two decimal places only
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
            return tl.models.Model(inputs=[s, a], outputs=x, name='Critic' + name)

        self.actor = get_actor([None, s_dim])
        print(self.actor)
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
        #("weights",self.actor.trainable_weights)
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

        self.loss_history.append((td_error, a_loss))
        #print("LOSS:",td_error, " ",a_loss)
        self.ema_update()

    def store_transition(self, s, a, r, s_):
        """
        Store data in data buffer
        :param s: state
        :param a: act
        :param r: reward
        :param s_: next state
        :return: None
        """

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

        tl.files.save_weights_to_hdf5(f'model/ddpg_actor_{LR_A}_{LR_C}.hdf5', self.actor)
        tl.files.save_weights_to_hdf5(f'model/ddpg_actor_target_{LR_A}_{LR_C}.hdf5', self.actor_target)
        tl.files.save_weights_to_hdf5(f'model/ddpg_critic_{LR_A}_{LR_C}.hdf5', self.critic)
        tl.files.save_weights_to_hdf5(f'model/ddpg_critic_target_{LR_A}_{LR_C}.hdf5', self.critic_target)

    def load_ckpt(self):
        """
        load trained weights
        :return: None
        """
        tl.files.load_hdf5_to_weights_in_order(f'model/ddpg_actor_{LR_A}_{LR_C}.hdf5', self.actor)
        tl.files.load_hdf5_to_weights_in_order(f'model/ddpg_actor_target_{LR_A}_{LR_C}.hdf5', self.actor_target)
        tl.files.load_hdf5_to_weights_in_order(f'model/ddpg_critic_{LR_A}_{LR_C}.hdf5', self.critic)
        tl.files.load_hdf5_to_weights_in_order(f'model/ddpg_critic_target_{LR_A}_{LR_C}.hdf5', self.critic_target)

    def get_latest_data(self):
        """
        Retrieve the latest 10 data sets from the database and perform calculations
        :return: None
        """
        conn = sqlite3.connect('processed_data.db')
        cursor = conn.cursor()

        cursor.execute("SELECT data FROM processed_data ORDER BY id DESC LIMIT 10")
        latest_data = cursor.fetchall()

        if not latest_data:
            return

        concatenated_data = np.concatenate([eval(data[0]) for data in latest_data])

        points = concatenated_data[:, :4]
        width = concatenated_data[:, 4]

        pen_to_center_distance = np.linalg.norm(points[:, :2] - points[:, 2:4], axis=1)

        new_data = np.column_stack((pen_to_center_distance, width))
        conn.close()

        increment = new_data[1:] - new_data[:-1]

        # distance decrease when width decrease
        decrease_ratio = len(increment[(increment[:, 1] <= 0) & (increment[:, 0] <= 0)]) / len(increment[(increment[:, 0] <= 0)])

        # distance decrease when width increase
        increase_ratio = len(increment[(increment[:, 1] <= 0) & (increment[:, 0] > 0)]) / len(increment[(increment[:, 0] > 0)])

        # distance increase when width decrease
        jitter_ratio = len(increment[(increment[:, 1] > 0) & (increment[:, 0] <= 0)]) / len(increment[(increment[:, 0] <= 0)])

        width_decrease_influence_prob = max(decrease_ratio ,0.01)

        width_increase_influence_prob = max(increase_ratio - jitter_ratio, 0.01)

        decrease_width_increase_distance = increment[(increment[:, 1] <= 0) & (increment[:, 0] > 0)]

        increase_width_increase_distance = increment[(increment[:, 1] > 0) & (increment[:, 0] > 0)]

        decrease_width_decrease_distance = increment[(increment[:, 1] <= 0) & (increment[:, 0] <= 0)]

        increase_width_decrease_distance = increment[(increment[:, 1] > 0) & (increment[:, 0] <= 0)]

        # 80% positions of decrease_width_increase_distance, describe the degree of user jitter
        jitter_factor = np.percentile(decrease_width_increase_distance[:, 0], 80) * jitter_ratio

        increase_factor_in = np.percentile(increase_width_increase_distance[:, 0], 80) # * (1 - width_increase_influence_prob)

        increase_factor_out = np.percentile(increase_width_decrease_distance[:, 0], 80) # * width_increase_influence_prob

        decrease_factor_in = np.percentile(decrease_width_decrease_distance[:, 0], 80) # * width_decrease_influence_prob

        decrease_factor_out = np.percentile(decrease_width_increase_distance[:, 0], 80) # * (1 - width_decrease_influence_prob)

        print(width_decrease_influence_prob,
                           width_increase_influence_prob,
                           jitter_factor,
                           increase_factor_in,
                           increase_factor_out,
                           decrease_factor_in,
                           decrease_factor_out)
        # update factors
        env.set_parameters(width_decrease_influence_prob,
                           width_increase_influence_prob,
                           jitter_factor,
                           increase_factor_in,
                           increase_factor_out,
                           decrease_factor_in,
                           decrease_factor_out)



if __name__ == '__main__':

    env = PathEnvironment()
    env = env.unwrapped

    # Define state space, action space, range of action magnitude
    # s_dim = env.observation_space.shape[0]
    state_shape = env.observation_space.shape
    s_dim = np.prod(state_shape)
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high

    # ddpg
    ddpg = DDPG(a_dim, s_dim, a_bound)

    ddpg.get_latest_data()

    # train：
    if args.train:  # train

        t0 = time.time()  # time

        # initialize graphic
        # plt.ion()
        # fig, axs = plt.subplots(3, 1, figsize=(10, 12))

        reward_history = []
        width_history = []

        conn = sqlite3.connect('processed_data.db')
        cursor = conn.cursor()

        cursor.execute("SELECT data FROM processed_data ORDER BY id DESC LIMIT 10")
        latest_data = cursor.fetchall()

        new_data = np.concatenate([eval(data[0]) for data in latest_data])
        print(len(new_data))
        conn.close()

        index = 0

        block_size = 3

        while index + block_size * 2 <= len(new_data):
            s = new_data[index:index + block_size]
            s_ = new_data[index + block_size:index + block_size * 2]

            a = s_[0, 4] - s[-1, 4]

            r = env.calculate_reward(s_)

            ddpg.store_transition(s, a, r / 10, s_)

            index += block_size

        for i in range(MAX_EPISODES):
            t1 = time.time()
            s = env.reset()
            ep_reward = 0

            for j in range(MAX_EP_STEPS):
                # Add exploration noise
                a = ddpg.choose_action(s)
                # To be able to keep the development going, here's another way to add exploration.
                # So need to need to create a normal distribution with a as the mean and VAR as the standard deviation, and then sample a from the normal distribution
                # Because a is the mean, the probability of a is maximized. But how big a is relative to the other probabilities by is adjusted by the VAR. Here we can actually add update the VAR to dynamically adjust the certainty of a
                # And then do the trimming
                # Or any suitable decay factor
                a = np.clip(np.random.normal(a, VAR), -2, 2)
                a = np.around(a, decimals=2)
                s_, r, done, info = env.step(a)
                ddpg.store_transition(s, a, r / 10, s_)

                N = min(15, s.shape[0])

                width_history.append(np.mean(s[-N:, 4]))

                # data full, start learning
                if ddpg.pointer > MEMORY_CAPACITY-len(new_data):
                    if ddpg.pointer == MEMORY_CAPACITY - len(new_data)+1:
                        print("\nDDPG starts to learn.")
                    ddpg.learn()

                # stroke_points = s_[:, :2]
                # path_centers = s_[:, 2:4]
                #
                # axs[0].plot(stroke_points[:, 0], stroke_points[:, 1],'r-')
                # axs[0].plot(path_centers[:, 0], path_centers[:, 1],'b-')
                #
                # plt.pause(0.1)

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
            if i == MAX_EPISODES - 1:
                with open(f'loss_history_{LR_A}_{LR_C}.txt', 'w') as f:
                    for td_error, a_loss in ddpg.loss_history:
                        f.write(f"Critic Loss: {td_error}, Actor Loss: {a_loss}\n")


            #     plt.show()
            #
            # if(i%MAX_EP_STEPS != 0):
            #     reward_history.append(ep_reward)

            # axs[1].clear()
            # axs[1].plot(reward_history, label='Reward')
            # axs[1].set_xlabel('Episode')
            # axs[1].set_ylabel('Cumulative Reward')
            # axs[1].legend()
            #
            # axs[2].clear()
            # axs[2].plot(width_history, label='Width')
            # axs[2].set_xlabel('Step')
            # axs[2].set_ylabel('Width')
            # axs[2].legend()

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
                    if j == MAX_EP_STEPS-1:
                        print(
                            '\rEpisode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                                i, MAX_EPISODES, ep_reward,
                                time.time() - t1
                            )
                        )



        print('\nRunning time: ', time.time() - t0)
        ddpg.save_ckpt()


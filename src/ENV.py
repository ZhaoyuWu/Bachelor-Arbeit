import gym
from gym import spaces
import numpy as np
import random
import math
from dtw import *
from scipy.spatial.distance import euclidean


class PathEnvironment(gym.Env):
    def __init__(self):
        super(PathEnvironment, self).__init__()

        # Definition of the State space and the Action space
        self.observation_space = spaces.Box(low=0, high=10, shape=(3, 5))    # 3x5 Matrix，3x(stroke horizontal coordinate, stroke vertical coordinate, path center horizontal coordinate, path center vertical coordinate, path width)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,))           # Continuous action [-1, 1]

        # Initializing
        self.state = np.zeros((3, 5))                                        # Zero Matrix
        self.avg_width = 3                                                   # Set the initial path width

        # Chang Factors ================================================================================================

        # Logic:
        #
        #                                                     - in scope (divergent)
        #                                   - width increased
        #                                                     - out of scope (convergent)
        #
        #                - user affected
        #
        #
        # Width changes                                       - in scope (convergent)
        #                                   - width decreased
        #                                                     - out of scope (convergent)
        #
        #                - user unaffected (+-jitter)

        # Width change influence factor

        self.width_decrease_influence_prob = 0.6
        # self.width_decrease_influence_prob = |convergent_points| / |total_points| (when width decreases)

        self.width_increase_influence_prob = 0.5
        # self.width_increase_influence_prob = |convergent_points| / |total_points| (when width increases)

        # Jitter factor

        self.jitter_factor = 0.15
        # self.jitter_factor = mean(get_increment(divergent_points))

        # Width change factors

        self.increase_factor_in = 0.2
        # Width increase factor in scope = mean(get_stroke_change(in_points)) (when width increases)

        self.increase_factor_out = -0.05
        # Width increase factor out of scope = mean(get_stroke_change(out_points)) (when width increases)

        self.decrease_factor_in = -0.1
        # Width decrease factor in scope = mean(get_stroke_change(in_points)) (when width decreases)

        self.decrease_factor_out = -0.05
        # Width decrease factor in scope = mean(get_stroke_change(in_points)) (when width decreases)

        # ==============================================================================================================

        self.stroke_direction = random.uniform(0, 2 * np.pi)              # Stroke start with pointing in a random direction
        self.path_direction = self.stroke_direction # random.uniform(0, 2 * np.pi)                # Path start with pointing in a random direction

        # Randomize the start of the stroke around the start of the path.
        radius = np.random.rand() * (self.avg_width / 2)
        angle = np.random.rand() * 2 * np.pi
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)

        self.prev_stroke_point = np.array([x, y])                            # Stroke start point
        self.prev_path_center = np.array([0.0, 0.0])                         # Path start point

    def reset(self):
        # reset state
        self.state = np.zeros((3, 5))
        return self.state

    def step(self, action):
        # Change of width: new width = old width + action
        width_change = action

        print("Action: ", action)

        old_width = self.avg_width
        self.avg_width += action

        # Sigmoid
        sigmoid_width = (self.avg_width - 1.5) * 2 / 5

        sigmoid_width = 1 / (1 + np.exp(-sigmoid_width))

        # Map (0,1) to (0.1,3)
        mapped_width = sigmoid_width * 3 + 0.1

        self.avg_width = mapped_width

        new_width = max(self.avg_width, 0.1)


        # print("Width: ",self.avg_width)

        new_state = np.zeros((3, 5))

        # Step size of stroke and path
        stroke_step_size = 0.05
        path_step_size = 0.05

        for i in range(3):
            # Update the direction of the path
            path_angle_change = np.random.uniform(-np.pi / 32, np.pi / 32)
            self.path_direction += path_angle_change

            # Update the center point of path
            path_direction = np.array([np.cos(self.path_direction), np.sin(self.path_direction)])
            path_center = self.prev_path_center + path_direction * path_step_size

            # Update the direction of stroke, it will follow the direction of the path.
            stroke_direction_adjustment = (self.path_direction - self.stroke_direction) * 0.1             # adjustment factor
            self.stroke_direction += stroke_direction_adjustment

            # Update the stroke point
            stroke_direction = np.array([np.cos(self.stroke_direction), np.sin(self.stroke_direction)])
            stroke_point = self.prev_stroke_point + stroke_direction * stroke_step_size

            distance = np.linalg.norm(path_center - stroke_point)
            # print("Distance: ",distance)

            # Width decreases
            if width_change < 0:
                if np.random.rand() < self.width_decrease_influence_prob:
                    # User affected when out of scope(threshold)
                    if ((old_width / 2) * 1.1 < distance):
                        adjustment_factor = width_change * self.decrease_factor_out                              # positive affected by the width decrease
                        stroke_point += (path_center - stroke_point) * adjustment_factor
                    # User affected when in scope(threshold)
                    else:
                        adjustment_factor = width_change * self.decrease_factor_in                               # positive affected by the width decrease
                        stroke_point += (path_center - stroke_point) * adjustment_factor
                else:
                    if (distance <= 7):  # Overflow prevention
                        jitter = np.random.uniform(0, self.jitter_factor)
                        adjustment_factor = width_change * jitter  # Jitter
                        stroke_point += (path_center - stroke_point) * adjustment_factor
                    else:
                        adjustment_factor = width_change * -0.1  # Jitter
                        stroke_point += (path_center - stroke_point) * adjustment_factor
            # Width increases
            else:
                if np.random.rand() < self.width_increase_influence_prob:
                    # User affected when out of scope (large)
                    if ((old_width / 2) * 1.1 < distance):
                        adjustment_factor = width_change * self.increase_factor_out                              # positive affected by the width increase
                        stroke_point += (path_center - stroke_point) * adjustment_factor
                        # User affected when in scope(small)
                    if ((old_width / 2) > distance):
                        adjustment_factor = width_change * self.increase_factor_in                               # negative affected by the width increase
                        stroke_point += (path_center - stroke_point) * adjustment_factor
                else:
                    if(distance <= 7):                                                                           # Overflow prevention
                        jitter = np.random.uniform(0, self.jitter_factor)
                        adjustment_factor = width_change * jitter                                                # Jitter
                        stroke_point += (path_center - stroke_point) * adjustment_factor
                    else:
                        adjustment_factor = width_change * -0.1  # Jitter
                        stroke_point += (path_center - stroke_point) * adjustment_factor


            # New states
            new_state[i, :2] = stroke_point
            new_state[i, 2:4] = path_center
            new_state[i, 4] = new_width

            # Save the newst location of path and stroke
            self.prev_stroke_point = stroke_point
            self.prev_path_center = path_center

        self.state = new_state

        # Calculate the reward
        reward = self.calculate_reward(self.state)
        done = False
        info = {}
        # print("selfstat: ",self.state)
        return self.state, reward, done, info

    def calculate_reward(self, state):
        # last 10 points are concerned
        # print("rewardstate: ",state)
        N = min(20, state.shape[0])
        stroke_points = state[-N:, 0:2]
        path_centers = state[-N:, 2:4]

        # Calculate the DTW distance between path and stroke
        dtw_result = dtw(stroke_points, path_centers, dist=euclidean)
        dtw_distance = dtw_result[0]
        alpha = 0.2
        reward_dtw_distance = np.exp(-alpha * dtw_distance)

        # width is averange of the last 10 widths
        width = np.mean(state[-N:, 4])
        # print("Width: ",width)
        epsilon = 0.001

        # Smaller width, larger reward
        reward_width = np.log((1 / (width + epsilon)) + 1)

        # If width >= distance
        if(width/2 >= dtw_distance+0.2):
            # weight of width and DTW distance
            w_width = 0.7
            w_dtw_distance = 0.3
        # If width < distance
        else:
            w_width = 0.2
            w_dtw_distance = 0.8

        total_reward = w_width * reward_width + w_dtw_distance * reward_dtw_distance
        # print("Width reward: ",w_width * reward_width, "\nDistance reward: ",w_dtw_distance * reward_dtw_distance)
        return total_reward


# register the environment
gym.register(id='PathEnv-v0', entry_point='path_env:PathEnvironment')



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
        self.avg_width = 9                                                   # Set the initial path width
        self.width_change_influence_prob = 0.8                               # width change influence probability


        self.stroke_direction = random.uniform(0, 2 * np.pi)              # Stroke start with pointing in a random direction
        self.path_direction = random.uniform(0, 2 * np.pi)                # Path start with pointing in a random direction

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
        self.avg_width += width_change
        new_width = self.avg_width

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

            if np.random.rand() < self.width_change_influence_prob:
                # If the width decreases, increase the probability or magnitude that the handwriting points are closer to the path's central axis
                if width_change < 0:
                    adjustment_factor = -width_change * 0.1                                               # positive affected by the width decrease
                    stroke_point += (path_center - stroke_point) * adjustment_factor
                else:
                    adjustment_factor = -width_change * 0.05                                              # positive affected by the width increase
                    stroke_point += (path_center - stroke_point) * adjustment_factor
            else:
                if width_change < 0:
                    adjustment_factor = -width_change * 0.2                                               # nagetive affected by the width decrease
                    stroke_point -= (path_center - stroke_point) * adjustment_factor
                else:
                    adjustment_factor = -width_change * 0.3                                               # nagetive affected by the width increase
                    stroke_point -= (path_center - stroke_point) * adjustment_factor

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

        return self.state, reward, done, info

    def calculate_reward(self, state):
        # last 10 points are concerned
        N = min(10, state.shape[0])
        stroke_points = state[-N:, 0:2]
        path_centers = state[-N:, 2:4]

        # Calculate the DTW distance between path and stroke
        dtw_result = dtw(stroke_points, path_centers, dist=euclidean)
        dtw_distance = dtw_result[0]
        alpha = 0.2
        reward_dtw_distance = np.exp(-alpha * dtw_distance)

        # width is averange of the last 10 widths
        width = np.mean(state[-N:, 4])
        epsilon = 0.001

        # Smaller width, larger reward
        reward_width = np.log((1 / (width + epsilon)) + 1)

        # weight of width and DTW distance
        w_width = 0.1
        w_dtw_distance = 0.9

        total_reward = w_width * reward_width + w_dtw_distance * reward_dtw_distance

        return total_reward


# register the environment
gym.register(id='PathEnv-v0', entry_point='path_env:PathEnvironment')



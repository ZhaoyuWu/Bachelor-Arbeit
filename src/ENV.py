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

        # 定义状态空间和动作空间
        self.observation_space = spaces.Box(low=0, high=10, shape=(3, 5))  # 3x5的矩阵，表示3个坐标点的状态信息，分别是笔迹点的横纵坐标，路径横纵坐标，路径宽度
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,))  # 动作空间是一个连续的区间

        # 初始化其他参数
        self.state = np.zeros((3, 5))  # 初始状态为全零矩阵
        self.avg_width = 9  # 假设初始路径的平均宽度为5
        self.width_change_influence_prob = 0.8 #宽度影响概率因子


        self.stroke_direction = random.uniform(0, 2 * np.pi)  # 笔迹初始随机方向
        self.path_direction = random.uniform(0, 2 * np.pi)  # 路径初始随机方向
        # 假设 self.avg_width 已经被定义
        radius = np.random.rand() * (self.avg_width / 2)  # 随机半径
        angle = np.random.rand() * 2 * np.pi  # 随机角度

        # 极坐标转笛卡尔坐标
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)

        self.prev_stroke_point = np.array([x, y]) # 上一个笔迹中心
        self.prev_path_center = np.array([0.0, 0.0])  # 上一个路径中心点

    def reset(self):
        # 重置环境到初始状态
        self.state = np.zeros((3, 5))  # 初始状态为全零矩阵
        return self.state

    def step(self, action):
        # 宽度变化
        width_change = action
        self.avg_width += width_change
        new_width = self.avg_width

        # 初始化新状态数组
        new_state = np.zeros((3, 5))

        # 设置笔迹和路径的步长
        stroke_step_size = 0.05
        path_step_size = 0.05

        for i in range(3):
            # 路径方向更新
            path_angle_change = np.random.uniform(-np.pi / 32, np.pi / 32)
            self.path_direction += path_angle_change

            # 更新路径中心点位置
            path_direction = np.array([np.cos(self.path_direction), np.sin(self.path_direction)])
            path_center = self.prev_path_center + path_direction * path_step_size

            # 笔迹方向更新，使其偏向路径方向
            stroke_direction_adjustment = (self.path_direction - self.stroke_direction) * 0.1  # 调整因子，使笔迹偏向路径方向
            self.stroke_direction += stroke_direction_adjustment

            # 更新笔迹点位置
            stroke_direction = np.array([np.cos(self.stroke_direction), np.sin(self.stroke_direction)])
            stroke_point = self.prev_stroke_point + stroke_direction * stroke_step_size

            if np.random.rand() < self.width_change_influence_prob:
                # 如果宽度减小，增加笔迹点向路径中轴靠拢的概率或幅度
                if width_change < 0:
                    adjustment_factor = -width_change * 0.1  # 根据宽度变化调整因子
                    stroke_point += (path_center - stroke_point) * adjustment_factor
                else:
                    adjustment_factor = -width_change * 0.05  # 根据宽度变化调整因子
                    stroke_point += (path_center - stroke_point) * adjustment_factor
            else:
                if width_change < 0:
                    adjustment_factor = -width_change * 0.2  # 根据宽度变化调整因子
                    stroke_point -= (path_center - stroke_point) * adjustment_factor
                else:
                    adjustment_factor = -width_change * 0.3  # 根据宽度变化调整因子
                    stroke_point -= (path_center - stroke_point) * adjustment_factor

            # 存储新状态
            new_state[i, :2] = stroke_point
            new_state[i, 2:4] = path_center
            new_state[i, 4] = new_width

            # 更新上一个笔迹点和路径中心点
            self.prev_stroke_point = stroke_point
            self.prev_path_center = path_center

        # 更新状态
        self.state = new_state

        # 计算奖励
        reward = self.calculate_reward(self.state)
        done = False
        info = {}

        return self.state, reward, done, info

    def calculate_reward(self, state):
        # 确保 N 不超过状态中的点的数量
        N = min(10, state.shape[0])

        # 提取笔迹点和路径中心点的最近N个点
        stroke_points = state[-N:, 0:2]
        path_centers = state[-N:, 2:4]

        # 使用 DTW 库计算这两个序列之间的距离，直接获取距离值
        dtw_result = dtw(stroke_points, path_centers, dist=euclidean)
        dtw_distance = dtw_result[0]  # 假设dtw返回的第一个元素是距离

        # 调整DTW距离奖励的计算方法，使其更敏感于距离的变化
        alpha = 0.2  # 控制指数函数衰减速度的参数
        reward_dtw_distance = np.exp(-alpha * dtw_distance)
        # 路径宽度取最近N个坐标点宽度的平均值
        width = np.mean(state[-N:, 4])
        epsilon = 0.001  # 用于避免除零错误的小值

        # 使用对数函数来放大宽度较小时的奖励，同时确保宽度越窄，奖励越高
        reward_width = np.log((1 / (width + epsilon)) + 1)

        # 调整奖励权重以反映新的评分标准
        w_width = 0.1  # 使宽度奖励的影响与DTW距离奖励相等
        w_dtw_distance = 0.9

        # 计算总奖励
        total_reward = w_width * reward_width + w_dtw_distance * reward_dtw_distance

        return total_reward


# 注册环境
gym.register(id='PathEnv-v0', entry_point='path_env:PathEnvironment')



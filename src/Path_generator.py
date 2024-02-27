import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class RandomPathGenerator:
    def __init__(self, num_steps=500):
        self.num_steps = num_steps
        self.path_direction = random.uniform(0, 2 * np.pi)
        self.prev_path_center = np.array([0.0, 0.0])
        self.prev_path_width = 1.0
        self.path = np.zeros((num_steps, 3))
        self.width_change = 0

    def generate_path(self):
        for i in range(self.num_steps):
            path_step_size = 0.1
            if i % 10 == 0:  # change width each 10 steps
                self.width_change = random.choice([-0.05, 0.05])
            path_width = max(1, min(4, self.prev_path_width + self.width_change))
            path_angle_change = np.random.uniform(-np.pi / 32, np.pi / 32)
            self.path_direction += path_angle_change
            path_direction = np.array([np.cos(self.path_direction), np.sin(self.path_direction)])
            path_center = self.prev_path_center + path_direction * path_step_size
            self.path[i] = np.array([path_center[0], path_center[1], path_width])
            self.prev_path_center = path_center
            self.prev_path_width = path_width
        return self.path

    def plot_path(self):
        fig, ax = plt.subplots()
        for i, point in enumerate(self.path):
            if i % 10 == 0:
                circle = patches.Circle((point[0], point[1]), point[2], edgecolor='yellow', facecolor='yellow')
                ax.add_patch(circle)
        plt.plot(self.path[:,0], self.path[:,1], '-o', markersize=2, linewidth=1, color='yellow')
        plt.axis('equal')
        plt.show()

if __name__ == "__main__":
    generator = RandomPathGenerator(num_steps=300)
    generator.generate_path()
    generator.plot_path()

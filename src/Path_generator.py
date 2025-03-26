import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class RandomPathGenerator:
    def __init__(self, num_steps=200, canvas_size=(40, 40)):
        self.num_steps = num_steps
        self.canvas_size = canvas_size
        self.path_direction = 0#random.uniform(0, 2 * np.pi)
        self.prev_path_center = np.array([0.0, 0.0])
        self.prev_path_width = 1.5
        self.path = np.zeros((num_steps, 3))  # [x, y, width]
        self.width_change = 0
        self.figure = 2          # Controls the generated graph types: 0 for random paths, 1 for polygons, and 2 for arcs.

    def generate_path(self):
        if self.figure == 0:
            for i in range(self.num_steps):
                path_step_size = 0.1
                if i % 10 == 0:  # change width each 10 steps
                    self.width_change = random.choice([-0.05, 0.05])
                #Controls the path width.
                path_width = 1.5 #max(0.5, min(2, self.prev_path_width + self.width_change))
                path_angle_change = np.random.uniform(-np.pi / 8, np.pi / 8)
                self.path_direction += path_angle_change
                path_direction = np.array([np.cos(self.path_direction), np.sin(self.path_direction)])
                path_center = self.prev_path_center + path_direction * path_step_size
                self.path[i] = np.array([path_center[0], path_center[1], path_width])
                self.prev_path_center = path_center
                self.prev_path_width = path_width
        elif self.figure == 1:
            for i in range(self.num_steps):
                path_step_size = 0.1
                if i % 50 == 0:  # change width each 10 steps
                    self.width_change = random.choice([-0.01, 0.01])
                # Controls the path width.
                path_width = 1.5#max(0.5, min(1.5, self.prev_path_width + self.width_change))
                if i % 100 == 0:
                    #path_angle_change = np.random.uniform(np.radians(115), np.radians(165))
                    path_angle_change = np.radians(90)
                else:
                    path_angle_change = 0
                self.path_direction += path_angle_change
                path_direction = np.array([np.cos(self.path_direction), np.sin(self.path_direction)])
                path_center = self.prev_path_center + path_direction * path_step_size
                self.path[i] = np.array([path_center[0], path_center[1], path_width])
                self.prev_path_center = path_center
                self.prev_path_width = path_width

        elif self.figure == 2:
            path_angle_change = np.random.uniform(np.pi / 128, np.pi / 140)
            for i in range(self.num_steps):
                path_step_size = 0.1
                if i % 10 == 0:  # change width each 10 steps
                    self.width_change = random.choice([-0.05, 0.05])
                # Controls the path width.
                path_width = 1.5#max(0.5, min(1.5, self.prev_path_width + self.width_change))
                self.path_direction += path_angle_change
                path_direction = np.array([np.cos(self.path_direction), np.sin(self.path_direction)])
                path_center = self.prev_path_center + path_direction * path_step_size
                self.path[i] = np.array([path_center[0], path_center[1], path_width])
                self.prev_path_center = path_center
                self.prev_path_width = path_width

        return self.path

    def plot_path(self):
        fig, ax = plt.subplots()
        ax.plot(self.path[:, 0], self.path[:, 1], '-o', markersize=2, linewidth=1, color='blue')
        plt.axis('equal')

        ax.set_xlim(-self.canvas_size[0] / 2, self.canvas_size[0] / 2)
        ax.set_ylim(-self.canvas_size[1] / 2, self.canvas_size[1] / 2)

        plt.show()

if __name__ == "__main__":
    generator = RandomPathGenerator(num_steps=200)
    path = generator.generate_path()
    print(path)
    generator.plot_path()

import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class RandomPathGenerator:
    def __init__(self, num_steps=500, canvas_size=(800, 800)):
        self.num_steps = num_steps
        self.canvas_size = canvas_size
        self.path_direction = random.uniform(0, 2 * np.pi)
        self.prev_path_center = np.array([0.0, 0.0])
        self.prev_path_width = 1.0
        self.path = np.zeros((num_steps, 3))  # [x, y, width]
        self.width_change = 0

    def generate_path(self):
        for i in range(self.num_steps):
            path_step_size = 0.1
            if i % 10 == 0:  # change width each 10 steps
                self.width_change = random.choice([-0.05, 0.05])
            path_width = 1.5 # max(0.5, min(2, self.prev_path_width + self.width_change))
            path_angle_change = np.random.uniform(-np.pi / 16, np.pi / 16)
            self.path_direction += path_angle_change
            path_direction = np.array([np.cos(self.path_direction), np.sin(self.path_direction)])
            path_center = self.prev_path_center + path_direction * path_step_size
            self.path[i] = np.array([path_center[0], path_center[1], path_width])
            self.prev_path_center = path_center
            self.prev_path_width = path_width

        # Scale and translate path to fit the canvas size
        # self.scale_and_translate_path()

        # self.boundary_restrictions()

        return self.path

    def boundary_restrictions(self):
        margin = 2
        min_x, max_x = np.min(self.path[:, 0]), np.max(self.path[:, 0])
        min_y, max_y = np.min(self.path[:, 1]), np.max(self.path[:, 1])

        path_center = np.array([(min_x + max_x) / 2, (min_y + max_y) / 2])
        canvas_center = np.array([self.canvas_size[0] / 2, self.canvas_size[1] / 2])

        offset = canvas_center - path_center

        for i in range(len(self.path)):
            self.path[i, 0] = np.clip(self.path[i, 0] + offset[0], margin, self.canvas_size[0] - margin)
            self.path[i, 1] = np.clip(self.path[i, 1] + offset[1], margin, self.canvas_size[1] - margin)


    def scale_and_translate_path(self):
        # Extract x, y coordinates
        x_coords = self.path[:, 0]
        y_coords = self.path[:, 1]

        # Compute scaling factors and translation offsets
        margin = 2
        canvas_width_with_margin = self.canvas_size[0] - 2 * margin
        canvas_height_with_margin = self.canvas_size[1] - 2 * margin

        x_scale = canvas_width_with_margin / (np.max(x_coords) - np.min(x_coords))
        y_scale = canvas_height_with_margin / (np.max(y_coords) - np.min(y_coords))
        scale_factor = min(x_scale, y_scale)

        x_offset = (canvas_width_with_margin - (
                    np.max(x_coords) - np.min(x_coords)) * scale_factor) / 2 + margin - np.min(x_coords) * scale_factor
        y_offset = (canvas_height_with_margin - (
                    np.max(y_coords) - np.min(y_coords)) * scale_factor) / 2 + margin - np.min(y_coords) * scale_factor

        # Apply scaling and translation to path coordinates
        self.path[:, 0] = self.path[:, 0] * scale_factor + x_offset
        self.path[:, 1] = self.path[:, 1] * scale_factor + y_offset

        self.path[:, 2] = self.path[:, 2] * scale_factor

    def plot_path(self):
        fig, ax = plt.subplots()
        for i, point in enumerate(self.path):
            circle = patches.Circle((point[0], point[1]), point[2] * 0.5, edgecolor='r', facecolor='none')
            print(point[0], point[1],point[2] * 0.5)
            ax.add_patch(circle)
        plt.plot(self.path[:,0], self.path[:,1], '-o', markersize=2, linewidth=1, color='blue')
        plt.axis('equal')
        plt.xlim(0, self.canvas_size[0])
        plt.ylim(0, self.canvas_size[1])
        plt.show()

if __name__ == "__main__":
    generator = RandomPathGenerator(num_steps=300)
    path = generator.generate_path()
    generator.plot_path()

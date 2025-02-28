import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

pattern = 'loss_history_*_*'

file_names = glob.glob(pattern)

lra_values = []
lrc_values = []
avg_values = []

for file_name in file_names:
    lra, lrc = file_name.split('_')[2:]
    lrc = lrc.split('.txt')[0]

    lra = float(lra)
    lrc = float(lrc)

    with open(file_name, 'r') as f:
        lines = f.readlines()
        actor_losses = []
        for line in lines:
            if "Actor Loss" in line:
                parts = line.split(",")
                for part in parts:
                    if "Actor Loss" in part:
                        actor_loss = float(part.split(":")[-1].strip())
                        actor_losses.append(actor_loss)

    avg_value = np.mean(actor_losses[:1000])

    lra_values.append(lra)
    lrc_values.append(lrc-lra)
    avg_values.append(avg_value)

lra_values = np.array(lra_values)
lrc_values = np.array(lrc_values)
avg_values = np.array(avg_values)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(lra_values, lrc_values, avg_values, cmap='viridis')

ax.set_xlabel('LRA')
ax.set_ylabel('LRC')
ax.set_zlabel('Average Actor Loss')

plt.show()

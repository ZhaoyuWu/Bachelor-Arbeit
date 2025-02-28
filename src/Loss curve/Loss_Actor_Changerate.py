import matplotlib.pyplot as plt
import numpy as np

learning_rates = ['1e-06', '1e-05', '0.0001', '0.001', '0.01', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1']
actor_losses_dict = {lr: [] for lr in learning_rates}

step = 5

# Collect actor losses from files
for lr in learning_rates:
    if lr == '1':
        lrc = 2 * int(lr)
    elif lr == '0.2':
        lrc = 0.3
    elif lr == '0.3':
        lrc = 0.4
    elif lr == '0.4':
        lrc = 0.5
    elif lr == '0.5':
        lrc = 0.6
    elif lr == '0.6':
        lrc = 0.7
    elif lr == '0.7':
        lrc = 0.8
    elif lr == '0.8':
        lrc = 0.9
    elif lr == '0.9':
        lrc = 1
    else:
        lrc = 2*float(lr)
    filename = f'loss_history_{lr}_{lrc}.txt'
    with open(filename, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if "Actor Loss" in line:
            parts = line.split(",")
            for part in parts:
                if "Actor Loss" in part:
                    actor_loss = -float(part.split(":")[-1].strip())
                    actor_losses_dict[lr].append(actor_loss)

# Compute derivatives as rate of change of actor loss with respect to iteration number
derivatives = [0] * (step + 1)
final_derivatives = []
for lr in learning_rates:
    actor_losses = actor_losses_dict[lr][:1000]
    for i in range(1 + step,1000):
        derivative = (actor_losses[i] - actor_losses[i-step]) / step  # Compute rate of change
        derivatives.append(derivative)
    final_derivatives.append(np.mean(derivatives))

# Plot derivatives
plt.figure(figsize=(10, 6))
plt.plot(learning_rates, final_derivatives, marker='o')
plt.xlabel('Learning Rate')
plt.ylabel('Rate of Change of Actor Loss')
plt.title('Rate of Change of Actor Loss with Respect to Learning Rate')
plt.grid(True)
plt.show()

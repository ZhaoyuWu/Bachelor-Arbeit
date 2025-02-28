import matplotlib.pyplot as plt

learning_rates = ['1e-06', '1e-05', '0.0001', '0.001', '0.01', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1']
#learning_rates = [ '0.1', '0.3', '0.2', '0.4', '0.6', '0.8']
actor_losses_dict = {lr: [] for lr in learning_rates}

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
                    actor_loss = float(part.split(":")[-1].strip())
                    actor_losses_dict[lr].append(actor_loss)

plt.figure(figsize=(10, 6))
for lr in learning_rates:
    plt.plot(actor_losses_dict[lr][:100], label=f'LR={lr}')

plt.xlabel('Iteration')
plt.ylabel('Actor Loss')
plt.title('Actor Loss Over First 500 Iterations for Different Learning Rates')
plt.legend()
plt.show()

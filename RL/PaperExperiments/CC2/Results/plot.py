import matplotlib.pyplot as plt
import pandas as pd
import os

# Directory containing the files
directory = "/scratch/egraham/CASTLE-GT/RL/CC2-implementation/Evaluation/plotting"

if not os.path.exists(directory):
    print(f"Directory does not exist: {directory}")
else:
    print(f"Directory found: {directory}")
    balances = []
    means = []
    std_devs = []

    # Iterate over each file in the directory
    for filename in sorted(os.listdir(directory)):
        if filename.startswith("B-PPOxCCE_balance") and filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            balance = int(filename.split('balance')[1].split('.')[0])

            with open(file_path, 'r') as file:
                lines = file.readlines()

            for line in lines:
                if line.startswith("steps: 30"):
                    parts = line.split(',')
                    mean = float(parts[2].split('mean: ')[1].strip())
                    std_dev = float(parts[3].split('standard deviation')[1].strip())
                    
                    balances.append(balance)
                    means.append(mean)
                    std_devs.append(std_dev)
                    break

    # DataFrame from the collected data
    data = pd.DataFrame({
        'Balance': balances,
        'Mean': means,
        'Standard Deviation': std_devs
    })

  
    data_sorted = data.sort_values(by='Balance')

    # Plotting RLxCCE Agent
    plt.figure(figsize=(10, 6))
    plt.errorbar(data_sorted['Balance'], data_sorted['Mean'], yerr=data_sorted['Standard Deviation'], fmt='o:', capsize=5, color='black', label='Agent-Agnostic EXP3-IX')

    # Mean Lines for purely PPO RL Agent (Red) and Extended (fully) Training RL Agent (Dark Green)
    rl_mean = -106.1025
    extended_rl_mean = -53.8897
    plt.axhline(y=rl_mean, color='blue', linestyle='dashed', linewidth=2, label='RL Agent')
    plt.axhline(y=extended_rl_mean, color='darkgreen', linestyle=':', linewidth=2, label='RL Agent x10 training')

    plt.title('Rewards vs CCE Certainty')
    plt.xlabel('Minimum observation-action visits (certainty) of CCE')
    plt.ylabel('Rewards (over 30 steps)')
    plt.grid(True)
    plt.legend(loc='lower right', fontsize='small')

    plt.savefig('/scratch/egraham/CASTLE-GT/RL/CC2-implementation/Results/plot-partial-2.png')
    print("Plot saved to CASTLE-GT/RL/CC2-implementation/Results/")
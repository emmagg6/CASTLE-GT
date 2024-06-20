'''
Plot all the data ... 
'''

import matplotlib.pyplot as plt
import pandas as pd
import os


costs = ['C0', 'C1', 'C2', 'C3', 'C4']
trn_intercals = ["1000", "2000", "3000", "4000", "5000", "6000", "7000", "8000", "9000", "10000"]
models = []

for trn_amnt in trn_intercals:
    for cost in costs:
        models.append(f'PPOxCCE_{cost}.txt')

        # Directory containing the files
        directory = f"/scratch/egraham/CASTLE-GT/RL/CC2-Cardiff-implementation/GT-Cardiff-CC2/Evaluation/{trn_amnt}/{cost}/"
        directory_ppo = f"/scratch/egraham/CASTLE-GT/RL/CC2-Cardiff-implementation/GT-Cardiff-CC2/Evaluation/{trn_amnt}/{cost}/ppo/"

        if not os.path.exists(directory):
            print(f"Directory does not exist: {directory}")
        else:
            print(f"Directory found: {directory}")
            balances = []
            means = []
            std_devs = []

            # Iterate over each file in the directory
            for filename in sorted(os.listdir(directory)):
                if filename.startswith("PPOxCCE_z") and filename.endswith(".txt"):
                    file_path = os.path.join(directory, filename)
                    balance = int(filename.split('z')[1].split('.')[0])

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

            for filename in sorted(os.listdir(directory_ppo)):
                if filename.startswith("PPOxCCE_z") and filename.endswith(".txt"):
                    file_path = os.path.join(directory_ppo, filename)
                    balance = int(filename.split('z')[1].split('.')[0])

                    with open(file_path, 'r') as file:
                        lines = file.readlines()

                    for line in lines:
                        if line.startswith("steps: 30"):
                            parts = line.split(',')
                            rl_mean = float(parts[2].split('mean: ')[1].strip())
                            rl_std_dev = float(parts[3].split('standard deviation')[1].strip())
                            
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
            plt.errorbar(data_sorted['Balance'], data_sorted['Mean'], yerr=data_sorted['Standard Deviation'], fmt='o:', capsize=5, color='black', label='EXP3-IXRL')

            # Mean Lines for purely PPO RL Agent (Red) and Extended (fully) Training RL Agent (Dark Green)
            plt.axhline(y=rl_mean, color='blue', linestyle='dashed', linewidth=2, label='RL Agent')
            plt.axhline(y=rl_mean + rl_std_dev, color='blue', linestyle=':', linewidth=2)
            plt.axhline(y=rl_mean - rl_std_dev, color='blue', linestyle=':', linewidth=2)


            plt.title('Rewards vs Certainty')
            plt.xlabel('Minimum observation-action visits (certainty measure) for the EXP3-IXRL approximation')
            plt.ylabel('Rewards (over 30 steps)')
            plt.grid(True)
            plt.legend(loc='lower right', fontsize='small')

            plt.savefig(directory + 'plot.png')
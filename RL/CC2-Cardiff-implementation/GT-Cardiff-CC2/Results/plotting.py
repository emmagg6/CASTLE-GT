'''
Plot all the data ... 
'''

import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap


costs = ['C0', 'C1', 'C2', 'C3', 'C4']
trn_intercals = ["1000", "2000", "3000", "4000", "5000", "6000", "7000", "8000", "9000", "10000"]
# trn_intercals = ["5000", "6000", "7000", "8000", "9000", "10000"]
models = []

for trn_amnt in trn_intercals:
    for cost in costs:
        models.append(f'PPOxCCE_{cost}.txt')

        # Directory containing the files
        directory = f"/scratch/egraham/CASTLE-GT/RL/CC2-Cardiff-implementation/GT-Cardiff-CC2/Evaluation/{trn_amnt}/{cost}/"

        if not os.path.exists(directory):
            print(f"Directory does not exist: {directory}")
        else:
            print(f"Directory found: {directory}")
            balances = []
            means = []
            std_devs = []
            proportions = []

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
                            proportion = float(parts[4].split('proportion CCE: ')[1].strip())

                            
                            if balance == 100000000000: 
                                rl_mean = mean
                                rl_std_dev = std_dev
                            else :
                                balances.append(balance)
                                means.append(mean)
                                std_devs.append(std_dev)
                                proportions.append(proportion)
                            break


            # DataFrame from the collected data
            data = pd.DataFrame({
                'Balance': balances,
                'Mean': means,
                'Standard Deviation': std_devs,
                'Percentage' : proportions
            })

            print("Dataframe:")
            print(data)

            print("RL Mean: ", rl_mean)
            print("RL Std Dev: ", rl_std_dev)

        
            data_sorted = data.sort_values(by='Balance')
            cmap = LinearSegmentedColormap.from_list('gray_to_black', ['gray', 'black'])
            norm = Normalize(vmin=0, vmax=1)
            colors = cmap(norm(data_sorted['Percentage']))

            # Plotting RLxCCE Agent
            plt.figure(figsize=(10, 6))

            # Mean Lines for purely PPO RL Agent (Blue)
            plt.axhline(y=rl_mean, color='blue', linestyle='dashed', linewidth=2, label='RL Agent', alpha=0.5, zorder=1)
            plt.axhline(y=rl_mean + rl_std_dev, color='blue', linestyle=':', linewidth=2, alpha=0.5, zorder=1)
            plt.axhline(y=rl_mean - rl_std_dev, color='blue', linestyle=':', linewidth=2, alpha=0.5, zorder=1)

            plt.errorbar(data_sorted['Balance'], data_sorted['Mean'], yerr=data_sorted['Standard Deviation'], fmt='o:', capsize=5, color='gray', label='EXP3-IXRL', markersize=1, zorder=1)

            plt.title('Rewards vs Certainty')
            plt.xlabel('Minimum observation-action visits (certainty measure) for the EXP3-IXRL approximation')
            plt.ylabel('Rewards (over 30 steps)')
            plt.grid(True)

            sm = ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            plt.colorbar(sm, ax=plt.gca(), label='Precentage Exp3-IXrl', shrink=0.7)

            # Scatter plot after other plot elements with increased zorder
            plt.scatter(data_sorted['Balance'], data_sorted['Mean'], c=colors, zorder=3)

            # Legend after scatter plot
            plt.legend(loc='lower right', fontsize='small')

            plt.savefig(directory + 'adhoc-plot.png')

            plt.close()

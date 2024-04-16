'''
This file takes in the rewards from the agents as saved in a .pth file that is exported before the memory is cleared during co-training.
The rewards are then plotted to visualize the learning dynamics of the agents.

Specify the agent's folder and whether the agent is 'Blue' or 'Red' in the main function.

'''

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys

def visualize_rewards(agent_folder, agent_name, window_size=1000):

    ckpt_folder = '/net/data/idsgnn/' + agent_folder # os.path.join(os.getcwd(), "TrainedModels", agent_folder)

    # load rewards
    rewards = torch.load(os.path.join(ckpt_folder, f'{agent_name}_rewards.pth'))
    time = np.arange(len(rewards))

    # calculate the moving average so see through the noise
    moving_avg = pd.Series(rewards).rolling(window=window_size, min_periods=1).mean()

    if agent_name == 'Blue':
        clr = 'b'
    else:
        clr = 'r'

    plt.figure(figsize=(10, 5))
    plt.plot(time, moving_avg, color=clr, label=f'{agent_name} ({agent_folder}) Moving Average (window={window_size})')
    # plt.scatter(time, rewards, color=clr, alpha=0.15, label='Raw Rewards')  # Optional: plot the raw rewards
    plt.xlabel('Timesteps')
    plt.ylabel('Average Reward')
    plt.title(f'{agent_name} Agent Rewards during Co-training')
    plt.legend()
    #plt.savefig('/scratch/egraham/CASTLE/TrainedModels/LearningDynamics/' + f'{agent_name}_{agent_folder}_rewards-windw{window_size}.png')
    plt.savefig('plots/' + f'{agent_name}_{agent_folder}_rewards-window{window_size}.png')
    print('Finished plotting')

    

folder = 'mini-test' # which co-trained agents
blue, red = 'Blue', 'Red' # select your agent

moving_avg_window = 10000 # for reference, window size for mini trained at 400 episodes was 100 

visualize_rewards(agent_folder=folder, agent_name=blue, window_size=moving_avg_window)

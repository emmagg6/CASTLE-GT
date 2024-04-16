# checkout https://github.com/geekyutao/PyTorch-PPO/blob/master/PPO_discrete.py

import torch
import numpy as np
import os
from CybORG import CybORG
from CybORG.Agents import BlueReactRemoveAgent
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2
from Wrappers.CoTraining import CoTraining
import inspect
from Agents.BlueAgents.BluePPOAgent import BluePPOAgent as BluePPOAgent
from Agents.RedAgents.RedPPOAgent import RedPPOAgent as RedPPOAgent

from CybORG.Agents import B_lineAgent
import functools as ft
from functools import partial
import random

PATH = str(inspect.getfile(CybORG))
PATH = PATH[:-10] + '/Shared/Scenarios/Scenario2.yaml'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':

    # set seeds for reproducibility
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    # change checkpoint directory
    # ckpt_folder = '/net/data/idsgnn/cotrain_both/'
    folder = "cotrain_both"
    ckpt_folder = ckpt_folder = os.path.join(os.getcwd(), "Models", folder)
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)

    ''' 
    print_interval = 50
    save_interval = 1  # episodes between checkpoints
    max_episodes = 1
    max_timesteps = 2000  # timesteps within an episode
    update_timesteps = 1000 # timesteps (state snapshots from memory) between PPO train() calls
    '''
    
    print_interval = 1 #50
    save_interval = 1000
    max_episodes = 100000
    max_timesteps = 100
    update_timesteps = 20000

    # PPO parameters
    K_epochs = 6
    eps_clip = 0.2
    gamma = 0.99
    betas=[0.9, 0.990]
    lr = 0.002

    co_train = CoTraining(
        print_interval= print_interval,
        save_interval= save_interval,
        max_episodes= max_episodes,
        max_timesteps= max_timesteps,
        update_timesteps= update_timesteps,
        K_epochs= K_epochs,
        eps_clip= eps_clip,
        lr=lr,
        gamma= gamma,
        betas= betas,
        ckpt_folder= ckpt_folder,
        scenario_path= PATH
    )

    # define ppo red agent
    input_dims_red = 40 # observation vector size for Red is 40
    start_actions_red=[]
    red_action_space=list(range(888))
    red_load_ckpt = None    # no Red checkpoint yet; Red is trained first against existing Blue
    red_restore = False
    if red_load_ckpt:
        red_restore = True
    red_agent = RedPPOAgent(input_dims_red, red_action_space, 
                            lr, betas, gamma, K_epochs, eps_clip, 
                            start_actions=start_actions_red,
                            ckpt=red_load_ckpt, 
                            restore=red_restore)
    co_train.add_agent(
        agent_name= "Red",
        agent = red_agent,
        action_space = red_action_space,
        start_actions = start_actions_red,
        agent_class = RedPPOAgent,
        load_ckpt = red_load_ckpt,
        input_dims = input_dims_red,
        restore = red_restore,
    )

    # define ppo blue agent
    blue_input_dims = 52  # size of the observation vector for blue
    start_actions_blue = [1004, 1004, 1000] # user 2 decoy * 2, ent0 decoy
    blue_action_space = [133, 134, 135, 139]  # restore enterprise and opserver
    blue_action_space += [3, 4, 5, 9]  # analyse enterprise and opserver
    blue_action_space += [16, 17, 18, 22]  # remove enterprise and opserer
    blue_action_space += [11, 12, 13, 14]  # analyse user hosts
    blue_action_space += [141, 142, 143, 144]  # restore user hosts
    blue_action_space += [132]  # restore defender
    blue_action_space += [2]  # analyse defender
    blue_action_space += [15, 24, 25, 26, 27]  # remove defender and user hosts
    blue_load_ckpt = None  
    #blue_load_ckpt = "Models/bline/model.pth"  # initial model that Red is trained against. It can also be None
    blue_restore = False
    if blue_load_ckpt:
        blue_restore = True
    blue_agent = BluePPOAgent(blue_input_dims, blue_action_space, 
                              lr, betas, gamma, K_epochs, eps_clip, 
                              start_actions=start_actions_blue,
                              ckpt=blue_load_ckpt,
                              restore=blue_restore) #need to think about restore
    co_train.add_agent(
        agent_name = "Blue",
        agent = blue_agent,
        action_space = blue_action_space,
        start_actions = start_actions_blue,
        agent_class = BluePPOAgent,
        load_ckpt = blue_load_ckpt,
        input_dims = blue_input_dims,
        restore = blue_restore,
    )

    # turn-based training
    co_train.cotrain()



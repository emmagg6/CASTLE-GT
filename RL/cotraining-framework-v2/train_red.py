# checkout https://github.com/geekyutao/PyTorch-PPO/blob/master/PPO_discrete.py

import torch
import numpy as np
import os
from CybORG import CybORG
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2
import inspect
from CybORG.Agents import BlueReactRemoveAgent
from Agents.RedAgents.RedPPOAgent import RedPPOAgent
from Agents.BlueAgents.BluePPOAgent import BluePPOAgent
import random
import functools as ft
from CybORG.Agents import B_lineAgent

PATH = str(inspect.getfile(CybORG))
PATH = PATH[:-10] + '/Shared/Scenarios/Scenario2.yaml'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(env, input_dims, action_space,
          max_episodes, max_timesteps, update_timestep, K_epochs, eps_clip,
          gamma, lr, betas, ckpt_in=None, ckpt_out=None, restore=False, print_interval=10, save_interval=100, start_actions=[]):

    agent_name = "Red"
    agent = RedPPOAgent(input_dims, action_space,
                              lr, betas, gamma, K_epochs, eps_clip,
                              ckpt=ckpt_in,
                              restore=restore, 
                              start_actions=start_actions)


    running_reward, time_step = 0, 0

    for i_episode in range(1, max_episodes + 1):
        state = env.reset()
        for t in range(max_timesteps):
            time_step += 1
            agent.time_step += 1

            agent.action_space_dict = env.get_action_space_dict(agent_name)
            agent.observation_dict = env.get_observation_dict(agent_name)

            action = agent.get_action(state)
            state, reward, done, _ = env.step(action)
            agent.store(reward, done)

            # reds_action = env.get_last_action('Red')
            #if str(reds_action) != "InvalidAction":
            #print(f"--External Red Agent Action: {env.get_last_action('Red')} timestep={t}")
            #print(f"--Internal Blue Agent Action: {env.get_last_action('Blue')} timestep={t}\n")

            if agent.time_step % update_timestep == 0:
                agent.train()
                
                avg_rewards = [
                        np.mean(agent.memory.rewards[-int(update_timestep/10):-int(2*update_timestep/10)]),
                        np.mean(agent.memory.rewards[-int(2*update_timestep / 10):-int(3*update_timestep/10)]),
                        np.mean(agent.memory.rewards[-int(3*update_timestep/10):-int(4*update_timestep/10)]),
                        np.mean(agent.memory.rewards[-int(4*update_timestep/10):-int(5*update_timestep/10)]),
                        np.mean(agent.memory.rewards[-int(5*update_timestep/10):-int(6*update_timestep/10)]),
                        np.mean(agent.memory.rewards[-int(6*update_timestep/10):-int(7*update_timestep/10)]),
                        np.mean(agent.memory.rewards[-int(7*update_timestep/10):-int(8*update_timestep/10)]),
                        np.mean(agent.memory.rewards[-int(8*update_timestep/10):-int(9*update_timestep/10)]),
                        np.mean(agent.memory.rewards[-int(9*update_timestep/10):]) ]

                agent.add_reward(avg_rewards)
                ckpt_rew = os.path.join(ckpt_out, f'Red_rewards.pth')
                agent.save_rewards(ckpt_rew)

                agent.clear_memory()
                time_step = 0
                agent.time_step = 0

            running_reward += reward

        agent.end_episode()

        if i_episode % save_interval == 0:
            ckpt = os.path.join(ckpt_out, '{}.pth'.format(i_episode))
            torch.save(agent.policy.state_dict(), ckpt)
            print('Checkpoint saved')

        if i_episode % print_interval == 0:
            running_reward = int((running_reward / print_interval))
            print('Episode {} \t Avg reward: {}'.format(i_episode, running_reward))
            running_reward = 0


if __name__ == '__main__':

    # set seeds for reproducibility
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    print_interval = 50
    save_interval = 1000
    max_episodes = 100000
    max_timesteps = 100
    # 200 episodes for buffer
    update_timesteps = 2000
    K_epochs = 6
    eps_clip = 0.2
    gamma = 0.99
    lr = 0.002
    betas=[0.9, 0.990]

    ckpt_folder = '/net/data/idsgnn/coRed_against_coBlue_35k_v3/'
    #ckpt_folder = '/net/data/idsgnn/coRed_against_BlueReactRemove/'
    #ckpt_folder = os.path.join(os.getcwd(), "Models", folder)
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)


    #CYBORG = CybORG(PATH, 'sim', agents={
    #    'Blue': BlueReactRemoveAgent
    #})


    # define ppo blue agent
    blue_start_actions = [1004, 1004, 1000] # user 2 decoy * 2, ent0 decoy
    blue_action_space = [133, 134, 135, 139]  # restore enterprise and opserver
    blue_action_space += [3, 4, 5, 9]  # analyse enterprise and opserver
    blue_action_space += [16, 17, 18, 22]  # remove enterprise and opserer
    blue_action_space += [11, 12, 13, 14]  # analyse user hosts
    blue_action_space += [141, 142, 143, 144]  # restore user hosts
    blue_action_space += [132]  # restore defender
    blue_action_space += [2]  # analyse defender
    blue_action_space += [15, 24, 25, 26, 27]  # remove defender and user hosts
    blue_ckpt = '/net/data/idsgnn/coblue_against_bline/35000.pth'

    CYBORG = CybORG(
            PATH,
            'sim',
            agents={
                "Blue": ft.partial(
                    BluePPOAgent,
                    input_dims = 52,
                    action_space = blue_action_space,
                    lr = lr,
                    betas = betas,
                    gamma = gamma,
                    K_epochs = K_epochs,
                    eps_clip = eps_clip,
                    start_actions = blue_start_actions,
                    ckpt = blue_ckpt,
                    restore = True,
                    internal = True,  # internal agent; will need to map the observation and the action to different format
                ),
            }
        )


    env = ChallengeWrapper2(env=CYBORG, agent_name="Red")
    #input_dims = env.observation_space.shape[0]

    start_actions = [] 
    action_space=list(range(888))
    input_dims = 40 # for red

    # define ppo red agent
    ckpt_out = ckpt_folder
    ckpt_in = None  
    restore = False
    if ckpt_in:
        restore = True
    
    train(env, input_dims, action_space,
              max_episodes=max_episodes, max_timesteps=max_timesteps,
              update_timestep=update_timesteps, K_epochs=K_epochs,
              eps_clip=eps_clip, gamma=gamma, lr=lr,
              betas=betas,
              print_interval=print_interval, ckpt_in=ckpt_in, ckpt_out=ckpt_out, restore=restore, save_interval=save_interval, start_actions=start_actions)



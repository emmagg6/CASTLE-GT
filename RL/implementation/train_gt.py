# checkout https://github.com/geekyutao/PyTorch-PPO/blob/master/PPO_discrete.py

import torch
import numpy as np
import os
from CybORG import CybORG
from CybORG.Agents import RedMeanderAgent, B_lineAgent, SleepAgent
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2
import inspect
from Agents.BlueAgents.GTAgent import GTAgent
from Agents.BlueAgents.ApproxCCEv2 import CCE
import random

PATH = str(inspect.getfile(CybORG))
PATH = PATH[:-10] + '/Shared/Scenarios/Scenario2.yaml'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(env, input_dims, action_space,
          max_episodes, max_timesteps, update_timestep, K_epochs, eps_clip,
          gamma, lr, betas, ckpt_folder, print_interval=10, save_interval=100, start_actions=[]):

    print('Training started')

    agent = GTAgent(input_dims, action_space, lr, betas, gamma, K_epochs, eps_clip, start_actions=start_actions)
    # cce = CCE(action_space=action_space)
    cce = CCE()

    running_reward, time_step = 0, 0

    # cce_log = []   # lst of [action, state, eq aprrox for state-action, count action selected for state, loss] lists

    for i_episode in range(1, max_episodes + 1):
        if i_episode % 10000 == 0 or i_episode < 10:
            print('Episode:', i_episode)
        if i_episode < 1501 and i_episode % 100 == 0:
            # print('Episode:', i_episode)
            # cce_t = cce_info[tuple(state)][0]
            # cce_visit_t = cce_info[tuple(state)][1]
            # print(f'Episode {i_episode} \t Avg reward: {running_reward/i_episode}, CCE approx {cce_t[action]:.3e} for state {state} and action {action} and visits {cce_visit_t[action]}')
            print(f'Episode {i_episode} \t Avg reward: {running_reward/i_episode} \t CCE approx {cce_info[tuple(state)][str(action)][0]:.3e} for state {state} and action {action} which has {cce_info[tuple(state)][str(action)][1]} visits')


        state = env.reset()
        for t in range(max_timesteps):
            time_step += 1
            action = agent.get_action(state)
            state, reward, done, _ = env.step(action)
            agent.store(reward, done)

            loss = agent.get_loss()
            cce_info = cce.update_eq(state, action, loss, T = max_episodes)

            if time_step % update_timestep == 0:
                agent.train()
                agent.clear_memory()
                time_step = 0

            running_reward += reward

        agent.end_episode()

        if i_episode % save_interval == 0:
            # cce_t = cce_info[tuple(state)][0]
            # cce_visit_t = cce_info[tuple(state)][1]
            # print(f'Episode {i_episode} \t Avg reward: {running_reward/i_episode} \t CCE approx {cce_t[action]:.3e} for state {state} and action {action} state visits {cce_visit_t[action]}')
            print(f'Episode {i_episode} \t Avg reward: {running_reward/i_episode} \t CCE approx {cce_info[tuple(state)][str(action)][0]:.3e} for state {state} and action {action} which has {cce_info[tuple(state)][str(action)][1]} visits')
            
            ckpt = os.path.join(ckpt_folder, '{}.pth'.format(i_episode))
            torch.save(agent.policy.state_dict(), ckpt)
            #---------------------
            cce.save(os.path.join(ckpt_folder, '{}cce.pkl'.format(i_episode)))
            #---------------------
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

    # change checkpoint directory
    folder = 'cce-bline'
    ckpt_folder = os.path.join(os.getcwd(), "Models", folder)
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)

    CYBORG = CybORG(PATH, 'sim', agents={
        # 'Red': B_lineAgent
        # 'Red': RedMeanderAgent
        'Red': SleepAgent
    })
    env = ChallengeWrapper2(env=CYBORG, agent_name="Blue")
    input_dims = env.observation_space.shape[0]

    action_space = [133, 134, 135, 139]     # restore enterprise and opserver
    action_space += [3, 4, 5, 9]            # analyse enterprise and opserver
    action_space += [16, 17, 18, 22]        # remove enterprise and opserer
    action_space += [11, 12, 13, 14]        # analyse user hosts
    action_space += [141, 142, 143, 144]    # restore user hosts
    action_space += [132]                   # restore defender
    action_space += [2]                     # analyse defender
    action_space += [15, 24, 25, 26, 27]    # remove defender and user hosts

    start_actions = [1004, 1004, 1000]      # user 2 decoy * 2, ent0 decoy

    print_interval = 10000 #50
    save_interval = 10000 #200
    max_episodes = 10000 # 100000
    max_timesteps = 100
    # 200 episodes for buffer
    update_timesteps = 20000
    K_epochs = 6
    eps_clip = 0.2
    gamma = 0.99
    lr = 0.002


    train(env, input_dims, action_space,
              max_episodes=max_episodes, max_timesteps=max_timesteps,
              update_timestep=update_timesteps, K_epochs=K_epochs,
              eps_clip=eps_clip, gamma=gamma, lr=lr,
              betas=[0.9, 0.990], ckpt_folder=ckpt_folder,
              print_interval=print_interval, save_interval=save_interval, start_actions=start_actions)
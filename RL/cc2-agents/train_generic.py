# checkout https://github.com/geekyutao/PyTorch-PPO/blob/master/PPO_discrete.py

import torch
import numpy as np
import os
import sys
from CybORG import CybORG
from CybORG.Agents import RedMeanderAgent, B_lineAgent, SleepAgent
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2
import inspect
from Agents.BlueAgents.PPOAgent import PPOAgent
from Agents.RedAgents.RandomAgent import RandomAgent
import random

PATH = str(inspect.getfile(CybORG))
PATH = PATH[:-10] + '/Shared/Scenarios/Scenario2Newx1000.yaml' # CHANGE FOLDER NAME line 132
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(envs, input_dims, action_space,
          max_episodes, max_timesteps, update_timestep, K_epochs, eps_clip,
          gamma, lr, betas, ckpt_folder, print_interval=10, save_interval=100, start_actions=[]):



    agent = PPOAgent(input_dims, action_space, lr, betas, gamma, K_epochs, eps_clip, start_actions=start_actions)

    running_reward, time_step = 0, 0

    for i_episode in range(1, max_episodes + 1):
        # For each episode, choose one of the environments to train against, alternating at each episode.
        env = envs[i_episode % len(envs)] 
        state = env.reset()
        for t in range(max_timesteps):
            time_step += 1
            action = agent.get_action(state)
            while True:
                # This is a workaround to a bug in CybORG that causes the random red agent to fail sometimes
                try:
                    state, reward, done, _ = env.step(action)
                except KeyError as e:
                    print(f'Bad action on Episode {i_episode}, step {t}, Action: {action}, Error: {e}. Re-rolling.')
                else:
                    break
            agent.store(reward, done)

            if time_step % update_timestep == 0:
                agent.train()
                agent.clear_memory()
                time_step = 0

            running_reward += reward

        agent.end_episode()

        if i_episode % save_interval == 0:
            ckpt = os.path.join(ckpt_folder, '{}.pth'.format(i_episode))
            torch.save(agent.policy.state_dict(), ckpt)
            print('Checkpoint saved')

        if i_episode % print_interval == 0:
            running_reward = int((running_reward / print_interval))
            print('Episode {} \t Avg reward: {}'.format(i_episode, running_reward))
            running_reward = 0


if __name__ == '__main__':
    try:
        max_episodes = int(sys.argv[1])
        print(f'num training episodes: {max_episodes}')
    except IndexError as e:
        raise IndexError(f'Must provide integer number of total episodes followed by a list of agents to train against (e.g. bline, meander, sleep, random) (original error: {e})')
    try:
        training_agents = sys.argv[2:]
        print(f'training uniformly against: {training_agents}')
    except IndexError as e:
        raise IndexError(f'Must provide number of total episodes followed by a list of agents to train against (e.g. bline, meander, sleep, random) (original error: {e})')

    # set seeds for reproducibility
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    envs = []
    # Make a training environment for each of the red agents
    for red_agent in training_agents:
        if red_agent == 'bline':
            red = B_lineAgent
        elif red_agent == 'meander':
            red = RedMeanderAgent
        elif red_agent == 'sleep':
            red = SleepAgent
        elif red_agent == 'random':
            red = RandomAgent
        else: raise NotImplementedError(f'{red_agent} not a valid red agent to train generic blue agent against')

        CYBORG = CybORG(PATH, 'sim', agents={
            'Red': red
        })
        envs = envs + [ChallengeWrapper2(env=CYBORG, agent_name="Blue")]


    # TODO: does this make a difference depending on the env??
    input_dims = envs[0].observation_space.shape[0]

    action_space = range(envs[0].get_action_space("Blue"))

    # action_space = [133, 134, 135, 139]  # restore enterprise and opserver
    # action_space += [3, 4, 5, 9]  # analyse enterprise and opserver
    # action_space += [16, 17, 18, 22]  # remove enterprise and opserer
    # action_space += [11, 12, 13, 14]  # analyse user hosts
    # action_space += [141, 142, 143, 144]  # restore user hosts
    # action_space += [132]  # restore defender
    # action_space += [2]  # analyse defender
    # action_space += [15, 24, 25, 26, 27]  # remove defender and user hosts

    # start_actions = [1004, 1004, 1000] # user 2 decoy * 2, ent0 decoy
    start_actions = []

    print_interval = 50
    save_interval = 200
    # max_episodes = 100000 # * len(envs) LISA NOTE: if training doesn't work very well, we should increase max_episodes to scale linearly with number of envs
    max_timesteps = 100
    # 200 episodes for buffer
    update_timesteps = 20000
    K_epochs = 6
    eps_clip = 0.2
    gamma = 0.99
    lr = 0.002
    
    # change checkpoint directory
    training_agents_str = '-'.join([a for a in training_agents])
    folder = f'generic_{training_agents_str}_{max_episodes}_x1000' 
    ckpt_folder = os.path.join(os.getcwd(), "Models", folder)
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)

    train(envs, input_dims, action_space,
              max_episodes=max_episodes, max_timesteps=max_timesteps,
              update_timestep=update_timesteps, K_epochs=K_epochs,
              eps_clip=eps_clip, gamma=gamma, lr=lr,
              betas=[0.9, 0.990], ckpt_folder=ckpt_folder,
              print_interval=print_interval, save_interval=save_interval, start_actions=start_actions)

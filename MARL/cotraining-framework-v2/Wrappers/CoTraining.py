import os 
import inspect
import functools as ft
import torch
import numpy as np
from CybORG import CybORG
from tqdm import tqdm
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2
from Agents.BlueAgents.BluePPOAgent import BluePPOAgent as BluePPOAgent
from Agents.RedAgents.RedPPOAgent import RedPPOAgent as RedPPOAgent


PATH = str(inspect.getfile(CybORG))
PATH = PATH[:-10] + '/Shared/Scenarios/Scenario2.yaml'


class CoTraining():
    def __init__(
            self, 
            print_interval,
            save_interval,
            max_episodes,
            max_timesteps,
            update_timesteps,
            K_epochs,
            eps_clip,
            gamma,
            lr,
            betas,
            ckpt_folder,
            scenario_path
        ):

        self.print_interval = print_interval
        self.save_interval = save_interval
        self.max_episodes = max_episodes
        self.max_timesteps = max_timesteps
        self.update_timesteps = update_timesteps
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.gamma = gamma
        self.lr = lr
        self.betas = betas
        self.ckpt_folder = ckpt_folder
        self.scenario_path = scenario_path
        print("log", self.scenario_path)
        self.train_agents = {}
    
    def add_agent(self, agent_name="", agent=None, action_space=[], start_actions=[], agent_class=None, load_ckpt="", input_dims=0, restore=False):
        self.train_agents[agent_name] = {
            "agent": agent,
            "action_space": action_space,
            "start_actions": start_actions,
            "agent_class": agent_class,
            "last_load_ckpt": load_ckpt,
            "input_dims": input_dims,
            "restore": restore,
        }
        if agent_name.lower() == "red":
            self.train_agents[agent_name]["ops"] = "Blue"
        else:
            self.train_agents[agent_name]["ops"] = "Red"

    def init_env(self, agent_name):
        ops = self.train_agents[agent_name]["ops"]
        CYBORG = CybORG(
            self.scenario_path,
            'sim',
            agents={
                ops: ft.partial(
                    self.train_agents[ops]["agent_class"],
                    input_dims = self.train_agents[ops]["input_dims"],
                    action_space = self.train_agents[ops]["action_space"],
                    lr = self.lr,
                    betas = self.betas,
                    gamma = self.gamma,
                    K_epochs = self.K_epochs,
                    eps_clip = self.eps_clip,
                    start_actions = self.train_agents[ops]["start_actions"],
                    ckpt = self.train_agents[ops]["last_load_ckpt"],
                    restore = self.train_agents[ops]["restore"],
                    internal = True,  # internal agent; will need to map the observation and the action to different format
                ),
            }
        )
        action_space_red = CYBORG.get_action_space('Red')
        action_space_blue = CYBORG.get_action_space('Blue')

        env = ChallengeWrapper2(env=CYBORG, agent_name=agent_name)
        return env, action_space_red, action_space_blue
    
    def cotrain(self):
        running_reward = 0

        for i_episode in range(1, self.max_episodes + 1):
            #if  i_episode % 1000 == 0:
            #print(f"Episode {i_episode}")
            # print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
            # print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
            # print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))

        ############ OPTIONAL DYNAMIC TIMESTEPS ############
            # self.max_timesteps = int((i_episode / (self.max_episodes / 1000))) + 5
            # self.update_timesteps = int(self.max_timesteps * 10) # this is just a thought, perhaps clear the mem in linear relation to the increasing training timesteps
    

            for agent_name in self.train_agents:
                #print("-------train", agent_name)

                agent = self.train_agents[agent_name]["agent"]
                env, action_space_red, action_space_blue = self.init_env(agent_name)
                state = env.reset()
                #print("A new env", state)
                
                # update the action space of the external_agent (it matters for red)
                #if agent_name == "Blue":
                #    agent.action_space_dict = action_space_blue
                #else:

                for t in range(self.max_timesteps):
                    agent.time_step += 1
                    
                    agent.action_space_dict = env.get_action_space_dict(agent_name)
                    agent.observation_dict = env.get_observation_dict(agent_name)
                    
                    action = agent.get_action(state)
                    state, reward, done, info = env.step(action)
                    
                    '''
                    if agent_name == 'Blue':
                        print(f"--External Blue Agent Action: {env.get_last_action('Blue')} timestep={t}")
                        print(f"--Internal Red Agent Action: {env.get_last_action('Red')} timestep={t}\n")
                    if agent_name == 'Red':
                        print(f"--External Red Agent Action: {env.get_last_action('Red')} timestep={t}")
                        print(f"--Internal Blue Agent Action: {env.get_last_action('Blue')} timestep={t}\n")
                    if str(env.get_last_action('Red')) == "InvalidAction": 
                        exit()

                    if str(env.get_last_action('Blue')) == "InvalidAction":
                        exit()
                    '''

                    #print(reward)
                    # print(info)

                    agent.store(reward, done)

                    if agent.time_step % self.update_timesteps == 0:
                        #print(f"Agent name {agent_name} is training")
                        agent.train()
                        ################ OPTIONAL FOR ANALYSIS ################
                        # before clearing memory, save / export the rewards
                        avg_rewards = [
                            np.mean(agent.memory.rewards[-int(self.update_timesteps/10):-int(2*self.update_timesteps/10)]), 
                            np.mean(agent.memory.rewards[-int(2*self.update_timesteps/10):-int(3*self.update_timesteps/10)]),
                            np.mean(agent.memory.rewards[-int(3*self.update_timesteps/10):-int(4*self.update_timesteps/10)]),
                            np.mean(agent.memory.rewards[-int(4*self.update_timesteps/10):-int(5*self.update_timesteps/10)]),
                            np.mean(agent.memory.rewards[-int(5*self.update_timesteps/10):-int(6*self.update_timesteps/10)]),
                            np.mean(agent.memory.rewards[-int(6*self.update_timesteps/10):-int(7*self.update_timesteps/10)]),
                            np.mean(agent.memory.rewards[-int(7*self.update_timesteps/10):-int(8*self.update_timesteps/10)]),
                            np.mean(agent.memory.rewards[-int(8*self.update_timesteps/10):-int(9*self.update_timesteps/10)]),
                            np.mean(agent.memory.rewards[-int(9*self.update_timesteps/10):]) ]

                        agent.add_reward(avg_rewards)
                        ckpt = os.path.join(self.ckpt_folder, f'{agent_name}_rewards.pth')
                        agent.save_rewards(ckpt)
                        #######################################################
                        agent.clear_memory()
                        agent.time_step = 0

                    
                    running_reward += reward

                agent.end_episode()

                if i_episode % self.save_interval == 0:
                    ckpt = os.path.join(self.ckpt_folder, f'{agent_name}_{i_episode}.pth')
                    agent.save(ckpt)
                    #print('Checkpoint saved')
                    self.train_agents[agent_name]["last_load_ckpt"] = ckpt
                    self.train_agents[agent_name]["restore"] = True

                if i_episode % self.print_interval == 0:
                    running_reward = int((running_reward / self.print_interval))
                    print(f'{agent_name} Episode {i_episode} \t Avg reward: {running_reward}')
                    running_reward = 0



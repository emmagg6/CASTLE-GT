import random
import torch
import copy,os
import numpy as np
import torch.nn as nn

from PPO.ActorCritic import ActorCritic
from PPO.Memory import Memory
from CybORG.Agents import BaseAgent
from CybORG.Shared import Results
from CybORG.Shared.Actions import PrivilegeEscalate, ExploitRemoteService, DiscoverRemoteSystems, Impact, \
    DiscoverNetworkServices, Sleep

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PPOAgent(BaseAgent):
    def __init__(self, input_dims=52, action_space=list(range(888)), lr=0.002, betas=[0.9, 0.990], gamma=0.99, K_epochs=4, eps_clip=0.2, restore=False, ckpt=None,
                 deterministic=False, training=True, start_actions=[], ckpt_dir=None, ckpt_old_dir=None):
        self.action = 0
        self.target_ip_address = None
        self.last_subnet = None
        self.last_ip_address = None
        self.action_history = {}
        self.jumps = [0,1,2,2,2,2,5,5,5,5,9,9,9,12,13]

        action_space = list(range(888))
        
        # PPO
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.input_dims = input_dims
        self.restore = restore
        self.ckpt = ckpt
        self.deterministic = deterministic
        self.training = training
        self.start = start_actions

        self.end_episode()
        # initialise
        self.set_initial_values(action_space=action_space)

        if ckpt_dir is not None:
            if os.path.exists(ckpt_dir):
                self.policy.load_state_dict(torch.load(ckpt_dir))
        if ckpt_old_dir is not None:
            if os.path.exists(ckpt_old_dir):
                self.old_policy.load_state_dict(torch.load(ckpt_old_dir))

    def save(self, path):
        torch.save(self.policy.state_dict(), path)
    
    def save_old(self, path):
        torch.save(self.old_policy.state_dict(), path)

    # concatenate the observation with the scan state
    def pad_observation(self, observation, old=False):
        if old:
            # added for store transition, remnants of DQN
            return np.concatenate((observation, self.scan_state_old))
        else:
            return np.concatenate((observation, self.scan_state))

    def store(self, reward, done):
        self.memory.rewards.append(reward)
        self.memory.is_terminals.append(done)

    def clear_memory(self):
        self.memory.clear_memory()

    def train(self):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.memory.rewards), reversed(self.memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        old_states = torch.squeeze(torch.stack(self.memory.states).to(device)).detach()
        old_actions = torch.squeeze(torch.stack(self.memory.actions).to(device)).detach()
        old_logprobs = torch.squeeze(torch.stack(self.memory.logprobs)).to(device).detach()

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            ratios = torch.exp(logprobs - old_logprobs.detach())

            advantages = rewards - state_values.detach()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = - torch.min(surr1, surr2)

            critic_loss = 0.5 * self.MSE_loss(rewards, state_values) - 0.01 * dist_entropy

            loss = actor_loss + critic_loss

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.old_policy.load_state_dict(self.policy.state_dict())



    # add scan information
    def add_scan(self, observation):
        # print(observation)
        indices = [0, 4, 8, 12, 28, 32, 36]#, 40, 44, 48] # TODO: what are the indices for the other scans?
        for id, index in enumerate(indices):
            # if scan seen on defender, enterprise 0-2, opserver0 or user 0-4
            if observation[index] == 1 and observation[index+1] == 0:
                # 1 if scanned before, 2 if is the latest scan
                self.scan_state = [1 if x == 2 else x for x in self.scan_state]
                self.scan_state[id] = 2
                break
            
            
    def get_action(self, obs, action_space=None, ret_probs=False):
        # print(self.action)
        
        # not needed for ppo since no transitions (remnant of DQNAgent)
        self.scan_state_old = copy.copy(self.scan_state)

        self.add_scan(obs)
        observation = self.pad_observation(obs)
        state = torch.FloatTensor(observation.reshape(1, -1)).to(device)
        action = self.old_policy.act(state, self.memory, deterministic=self.deterministic, ret_probs=ret_probs)
        if ret_probs:
            action, action_probs = action
        action_ = self.action_space[action]

        # force start actions, ignore policy. only for training
        if len(self.start_actions) > 0:
            action_ = self.start_actions[0]
            self.start_actions = self.start_actions[1:]

        if ret_probs:
            return action_, action_probs
        return action_


        # session = 0

        # while True:
        #     # TODO: seems no 'success' in observation
        #     # if observation['success'] == True:
        #     #     self.action += 1 if self.action < 14 else 0
        #     # else:
        #     #     self.action = self.jumps[self.action]

        #     # if observation['success'] is not True:
        #     #     self.action = self.jumps[self.action]

        #     if self.action in self.action_history:
        #         action = self.action_history[self.action]

        #     # Discover Remote Systems
        #     elif self.action == 0:
        #         self.initial_ip = observation['User0']['Interface'][0]['IP Address']
        #         self.last_subnet = observation['User0']['Interface'][0]['Subnet']
        #         action = DiscoverRemoteSystems(session=session, agent='Red', subnet=self.last_subnet)
        #     # Discover Network Services- new IP address found
        #     elif self.action == 1:
        #         hosts = [value for key, value in observation.items() if key != 'success']
        #         get_ip = lambda x : x['Interface'][0]['IP Address']
        #         interfaces = [get_ip(x) for x in hosts if get_ip(x)!= self.initial_ip]
        #         self.last_ip_address = random.choice(interfaces)
        #         action =DiscoverNetworkServices(session=session, agent='Red', ip_address=self.last_ip_address)

        #     # Exploit User1
        #     elif self.action == 2:
        #          action = ExploitRemoteService(session=session, agent='Red', ip_address=self.last_ip_address)

        #     # Privilege escalation on User Host
        #     elif self.action == 3:
        #         hostname = [value for key, value in observation.items() if key != 'success' and 'System info' in value][0]['System info']['Hostname']
        #         action = PrivilegeEscalate(agent='Red', hostname=hostname, session=session)

        #     # Discover Network Services- new IP address found
        #     elif self.action == 4:
        #         self.enterprise_host = [x for x in observation if 'Enterprise' in x][0]
        #         self.last_ip_address = observation[self.enterprise_host]['Interface'][0]['IP Address']
        #         action = DiscoverNetworkServices(session=session, agent='Red', ip_address=self.last_ip_address)

        #     # Exploit- Enterprise Host
        #     elif self.action == 5:
        #         self.target_ip_address = [value for key, value in observation.items() if key != 'success'][0]['Interface'][0]['IP Address']
        #         action = ExploitRemoteService(session=session, agent='Red', ip_address=self.target_ip_address)

        #     # Privilege escalation on Enterprise Host
        #     elif self.action == 6:
        #         hostname = [value for key, value in observation.items() if key != 'success' and 'System info' in value][0]['System info']['Hostname']
        #         action = PrivilegeEscalate(agent='Red', hostname=hostname, session=session)

        #     # Scanning the new subnet found.
        #     elif self.action == 7:
        #         self.last_subnet = observation[self.enterprise_host]['Interface'][0]['Subnet']
        #         action = DiscoverRemoteSystems(subnet=self.last_subnet, agent='Red', session=session)

        #     # Discover Network Services- Enterprise2
        #     elif self.action == 8:
        #         self.target_ip_address = [value for key, value in observation.items() if key != 'success'][2]['Interface'][0]['IP Address']
        #         action = DiscoverNetworkServices(session=session, agent='Red', ip_address=self.target_ip_address)

        #     # Exploit- Enterprise2
        #     elif self.action == 9:
        #         self.target_ip_address = [value for key, value in observation.items() if key != 'success'][0]['Interface'][0]['IP Address']
        #         action = ExploitRemoteService(session=session, agent='Red', ip_address=self.target_ip_address)

        #     # Privilege escalation on Enterprise2
        #     elif self.action == 10:
        #         hostname = [value for key, value in observation.items() if key != 'success' and 'System info' in value][0]['System info']['Hostname']
        #         action = PrivilegeEscalate(agent='Red', hostname=hostname, session=session)

        #     # Discover Network Services- Op_Server0
        #     elif self.action == 11:
        #         action = DiscoverNetworkServices(session=session, agent='Red', ip_address=observation['Op_Server0']['Interface'][0]['IP Address'])

        #     # Exploit- Op_Server0
        #     elif self.action == 12:
        #         info = [value for key, value in observation.items() if key != 'success']
        #         if len(info) > 0:
        #             action = ExploitRemoteService(agent='Red', session=session, ip_address=info[0]['Interface'][0]['IP Address'])
        #         else:
        #             self.action = 0
        #             continue
        #     # Privilege escalation on Op_Server0
        #     elif self.action == 13:
        #         action = PrivilegeEscalate(agent='Red', hostname='Op_Server0', session=session)
        #     # Impact on Op_server0
        #     elif self.action == 14:
        #         action = Impact(agent='Red', session=session, hostname='Op_Server0')

        #     if self.action not in self.action_history:
        #         self.action_history[self.action] = action
        #     return action

    def end_episode(self):
        self.action = 0
        self.target_ip_address = None
        self.last_subnet = None
        self.last_ip_address = None
        self.action_history = {}

        # TODO
        # 10 possible scans: defender, enterprise 0-2, user 0-4, opserver
        self.scan_state = np.zeros(10)
        # remnants of DQNAgent for store_transitions
        self.scan_state_old = np.zeros(10)
        # add start actions
        self.start_actions = copy.copy(self.start)




    def set_initial_values(self, action_space, observation=None):
        self.memory = Memory()

        self.n_actions = len(action_space)
        self.action_space = action_space 

        # add 10 to input_dims for the scanning state
        self.input_dims += 10


        self.policy = ActorCritic(self.input_dims, self.n_actions).to(device)
        if self.restore:
            pretained_model = torch.load(self.ckpt, map_location=lambda storage, loc: storage)
            self.policy.load_state_dict(pretained_model)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=self.betas)

        self.old_policy = ActorCritic(self.input_dims, self.n_actions).to(device)
        self.old_policy.load_state_dict(self.policy.state_dict())

        self.MSE_loss = nn.MSELoss()
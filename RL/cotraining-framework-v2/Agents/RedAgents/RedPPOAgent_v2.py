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

from Wrappers.RedVectorize import RedVectorize
from pprint import pprint

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RedPPOAgent(BaseAgent):
    def __init__(self, input_dims=50, action_space=list(range(888)), lr=0.002, betas=[0.9, 0.990], gamma=0.99, K_epochs=4, eps_clip=0.2, start_actions=[], restore=False, ckpt=None,
                 deterministic=False, training=True, time_step = 0, rewards = [], possible_actions = [], internal=False):
        
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
        self.start_actions = start_actions
        self.time_step = time_step
        self.rewards = rewards
        self.possible_actions = possible_actions
        self.internal = internal
        
        self.action_space=list(range(888))
        self.redVect = RedVectorize()
        self.last_action = None # need last_action in RedVectorize; the observation is updated based on it

        self.set_initial_values_train(action_space = self.action_space)

    def save(self, path):
        print("----Red saves policy to path ", path)
        torch.save(self.policy.state_dict(), path)
    
    def save_old(self, path):
        torch.save(self.old_policy.state_dict(), path)

    def store(self, reward, done):
        # print("Red stores: ", reward, done)
        self.memory.rewards.append(reward)
        self.memory.is_terminals.append(done)

    def save_rewards(self, path):
        # Check if the rewards file already exists
        if os.path.isfile(path):
            existing_rewards = torch.load(path, map_location=lambda storage, loc: storage) # existing rewards
            updated_rewards = existing_rewards + self.rewards # add the new rewards to the existing rewards
        else:
            # If the file does not exist, just use the new rewards
            updated_rewards = self.rewards
        torch.save(updated_rewards, path)

    def add_reward(self, rewards):
        self.rewards = self.rewards + rewards

    def clear_memory(self):
        self.memory.clear_memory()

#    def train(self):
    def train(self, results=None):  # the cyborg format requires the results parameter
        print("I am training in RedPPOAgent class:", self)
        print(results)
        """allows an agent to learn a policy"""
        if self.internal:
            return

        print("Training external RedPPO")
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


    def get_action(self, obs, action_space=None):
        print("Red's action space:", action_space)
        #print("Red Observation:")
        #pprint(obs)

        if self.internal:  # internal to the cyborg environment; need to convert to the vector representation
            obs = self.redVect.observation_change(obs, self.last_action)
            #print("Red Vect Observation:", obs)

        state = torch.FloatTensor(obs.reshape(1, -1)).to(device)
        action = self.old_policy.act(state, self.memory, deterministic=self.deterministic)
        
        action_ = self.action_space[action]

        #print("xxxxxxFinished RedPPO get_action, chose:", action_, self.possible_actions[action_])
        #print("xxxxRed's possible actions:", self.possible_actions)
        if self.internal:  # internal to the cyborg environment; need to convert to the correct action object
            # self.last_action = self.possible_actions[action_]
            return self.possible_actions[action_]

        return action_

    '''
    def test_valid_action(self, action: Action):
        # returns true if the parameters in the action are in and true in the action set else return false
        action_space = agent.action_space.get_action_space()

        print('test_valid_action, action_space: ', action_space['action'])
        print(action, agent)

        # first check that the action class is allowed
        if type(action) not in action_space['action'] or not action_space['action'][type(action)]:
            print("inval1")
            return False

        # next for each parameter in the action
        for parameter_name, parameter_value in action.get_params().items():
            print("parameter_name, parameter_value:", parameter_name, parameter_value)
            if parameter_name not in action_space:
                continue
            if parameter_value not in action_space[parameter_name] or not action_space[parameter_name][parameter_value]:
                print("action_space[parameter_name]:", action_space[parameter_name])
                print("action_space[parameter_name][parameter_value]:", action_space[parameter_name][parameter_value])
                print("inval2")
                return False
        return True
    '''

    # used in the EnvironmentControllet.py 
    def set_last_action(action):
        self.last_action = action
        print("Environm is setting last action to ", self.last_action)

    def end_episode(self):

        # add start actions
        self.start_actions = []

    def set_initial_values(self, action_space, observation=None):
        pass

    def set_initial_values_train(self, action_space, observation=None):
        self.memory = Memory()
        self.n_actions = len(self.action_space)
        self.policy = ActorCritic(self.input_dims, self.n_actions).to(device)

        #print('Restore:', self.restore)
        #print('ckpt dir: ', self.ckpt)

        if self.restore:
            print("Restoring Red from", self.ckpt, 'internal', self.internal)
            pretained_model = torch.load(self.ckpt, map_location=lambda storage, loc: storage)
            self.policy.load_state_dict(pretained_model)
        #else:
        #    print("Not Restoring Red,internal", self.internal)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=self.betas)

        self.old_policy = ActorCritic(self.input_dims, self.n_actions).to(device)
        self.old_policy.load_state_dict(self.policy.state_dict())

        self.MSE_loss = nn.MSELoss()

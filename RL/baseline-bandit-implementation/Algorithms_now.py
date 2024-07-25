import numpy as np
from typing import Callable



class EXP3IXrl(Algorithm): 
    ''' Implementation of the EXP3-IXrl algorithm ''' 
    def __init__(self, rl: str, time_horizon: int, gamma: float = None, dynamic_eta: bool = True): 
        ''' Initialize the EXP3-IXrl algorithm. ''' 
        self.n = 0 
        self.time_horizon = time_horizon  # For g = T in gamma upper bound on G_max in the paper 
        self.t = 0 # start at time 0 
        self.weights = {} # dictionary of weights for each action 
        self.action_probs = {} # dictionary of action probabilities 
        self.action_selected = {} # dictionary of the number of times the actions are sampled / selected  
        self.eta = np.sqrt((2 * np.log(self.n))/(self.n * self.time_horizon))  
        self.gamma = self.eta / 2  
        self.dynamic_eta = dynamic_eta   
        
    def train_step(self, action: int, reward: float): 
        ''' Train the EXP3 algorithm.  ''' 
        self.t += 1 
        a = str(action)  
        if a not in self.weights: 
            self.weights[a] = 1 
            self.action_probs[a] = 1/self.n  
        if self.dynamic_eta: 
            self.eta = np.sqrt((2 * np.log(self.n))/(self.n * self.t)) 
            self.gamma = self.eta / 2  # Make sure the loss which is l=-r is in [0, 1] 
            
        estimated_loss = (-1 * reward + 1) / (self.action_probs[action] + self.gamma) # l_t / (p_t(a) + γ)  
        # estimated_losses_vector = np.eye(self.n)[action] * estimated_loss  
        # # self.weights = self.weights * np.exp(-1 * self.eta * estimated_losses_vector)  
        self.weights[a] *= np.exp(-1 * self.eta * estimated_loss - np.max(list(self.weights.values())))  
        
    def get_equilibrium(self): 
        return self.weights, self.action_probs  
    
    def select_action(self, eq, visits, certainty) -> int: 
        # only select actions have been seen are above the certainty threshold  
        valid_actions = [a for a in visits.keys() if visits[a] > certainty] 
        
        if len(valid_actions) == 0: 
            return -1 
        else: 
            # chose the highest weights out of the the valid actions 
            return max(valid_actions, key=lambda x: eq[str(x)])  
        
        
    def reset(self) -> None: 
        ''' Reset the weights. ''' 
        self.weights = {} 
        self.action_probs = {} 
        self.action_selected = {} 
        self.t = 0  
        
        
    def __str__(self) -> str: return f'EXP3IX'
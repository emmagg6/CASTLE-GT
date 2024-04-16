import numpy as np
import torch
import pickle


class CCE():
    def __init__(self):
        self.cce = {}                     # dict for state to a specific equilibrium approximation and visit count for each action

    def update_eq(self, state, action, loss, 
                  action_space=[i for i in range(158)], eta=0.1, gamma=0.01): 
        '''
            Appros EQ update using the EXP3-IX algorithm

            Inputs:
                state: the current state
                action: the action taken
                valid_actions: list of valid actions for the state
                loss: the loss for the action
                EXP3-IX parameters: eta, gamma

            Updates:
                New state: initializes the state in EQ approximation and visit count dicts
                self.cce: the policy and count for the respective state and action
        '''
        s = tuple(state)            # to enable mutability of dict keys
        A = len(action_space)       # number of actions
        a = action                  # an integer representing the action

        current_approx = {}   # current equilibrium approximation
        current_count = {}    # current visit count

        if s not in self.cce:
            self.cce[s] = [np.full(A, 1.0 / A)]    # initializes as a uniform dist. 
            self.cce[s].append(np.zeros(A))        # initializes as a zero count

        # loss passed in as tensor with grad after initial iteration typically
        if isinstance(loss, torch.Tensor):
            loss = loss.detach().numpy() if loss.requires_grad else loss.numpy()
        loss = np.sum(loss)


        [current_approx[s], current_count[s]] = self.cce[s]

        # estimate loss
        estimated_loss = loss / (current_approx[s][a] + 1e-50) + gamma

        # Update eq and count
        current_approx[s][a] *= np.exp(-eta * estimated_loss)
        current_approx[s] /= np.sum(current_approx[s])
        current_count[s][a] += 1

        self.cce[s] = [current_approx[s], current_count[s]]  # Update the tuple in the dictionary

        return self.cce


    def save(self, path):
        '''
            Save the eq approximations to a file with pickle
        '''
        # saving cce with pickle since torch.load is not working for such a large file
        with open(path, 'wb') as f:
            pickle.dump(self.cce, f)

    
    def get_eq_action_visits(self, observation):
        '''
            Get the action based on the eq approximations
        '''
        current_approx, current_count = self.cce[observation]
        action = np.argmax(current_approx)
        return action, current_count[observation][action]
    
    def load_eq(self, path):
        '''
            Load the eq approximations from a file
        '''
        # self.eq_approx = torch.load(path)
        # return self.eq_approx
        # dict too large for torch.load -- use numpy instead
        self.cce = np.load(path, allow_pickle=True)
        return self.cce
    
    




'''
If a bunch of running information is still printing out, go to :
cage-challege-2-main/CybORG/CybORG/Simulator/State.py
and comment out the print statements that are uncommented 

-- updates to this file are not being pushed to the main branch
'''
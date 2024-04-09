import numpy as np
import torch


class CCE():
    def __init__(self):
        self.eq_approx = {}                     # dict for state to a specific equilibrium approximation
        self.visit_count = {}                   # dict for state to a visit count np.array


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
                self.eq_approx: the policy for the respective state and action
                self.as_count: the visit count for the respective state and action
        '''
        s = tuple(state)            # to enable mutability of dict keys
        A = len(action_space)       # number of actions
        a = action                  # an integer representing the action


        if s not in self.eq_approx:
            # self.eq_approx[s] = np.full(A, 1.0 / A)    # initializes as a uniform dist. 
            self.eq_approx[s] =  np.full(A, 1.0)         # to initialize starting at 1.0 (as Ryan descibed)
            self.visit_count[s] = np.zeros(A)  

        # estimate loss
        estimated_loss = loss / self.eq_approx[s][a] + gamma

        # update eq and count
        self.eq_approx[s][a] *= np.exp(-eta * estimated_loss)
        self.eq_approx[s] /= np.sum(self.eq_approx[s])
        self.visit_count[s][a] += 1

        return self.eq_approx[s][a], self.visit_count[s][a] # for eval or to track


    def eq_save(self, path):
        '''
            Save the eq approximations to a file
        '''
        torch.save(self.eq_approx, path)

    def get_counts(self):
        '''
            Return the visit counts
        '''
        return self.visit_count
    


'''
If a bunch of running information is still printing out, go to :
cage-challege-2-main/CybORG/CybORG/Simulator/State.py
and comment out the print statements that are uncommented 

-- updates to this file are not being pushed to the main branch
'''
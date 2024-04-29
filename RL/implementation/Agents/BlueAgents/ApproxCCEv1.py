import numpy as np
import torch
import pickle

class CCE():
    def __init__(self, action_space=[i for i in range(150)]):
        '''
            EQ Approximation class that implements the EXP3-IX algorithm

            Initializes:
                self.cce: dictionary tracks a str-state to a specific:
                  equilibrium approximation and visit count for each action

            Note: different take on EXP3-IX algorithm - this uses the log probabilities to enable scaling
            '''
        self.cce = {}
        self.eq = {}
         # action space passed to PPO but then added the 3 decoy actions, then actions that popped up as being used/tried during training outside of the action space argument
        self.action_space = action_space + [51] + [116] + [55] + [37, 61, 43, 130, 91, 44, 38, 115, 54, 107, 76, 131, 106, 28, 35, 120, 90, 119, 102, 29, 113, 126]
        self.action_index = {action: i for i, action in enumerate(self.action_space)}


    def initialize_or_update_cce(self, state, action):
        s = tuple(state)
        A = max(self.action_space) + 1
        a = action

        #-------------------------------- New State
        if s not in self.cce:
            self.cce[s] = [np.zeros(A, dtype=np.float64), np.zeros(A, dtype=np.float64)]

            for action in self.action_space:
                self.cce[s][0][action] = 1.0       # initializes as a one for valid actions

        #-------------------------------- New (valid) Action
                # In case some decoy actions are not in the action space -- should not be but curious
        if a not in self.action_space:
            print("Action not in action space: ", a)
            self.action_space.append(a)
            self.action_index[a] = len(self.action_space) - 1
            A = max(self.action_space) + 1

            for key in self.cce:
                current_length = len(self.cce[key][0])
                if a >= current_length:
                    print("Action not in action space: ", a)
                    extension_length = a - current_length + 1  # +1 to include the index_a itself

                    # extending the eq approx and visit count arrays with zeros up to the index of 'a'
                    self.cce[key][0] = np.pad(self.cce[key][0], (0, extension_length), 'constant', constant_values=(0.0)).astype(np.float64)
                    self.cce[key][1] = np.pad(self.cce[key][1], (0, extension_length), 'constant', constant_values=(0.0)).astype(np.float64)

                # Set the specific indices to desired values - initialize the eq approx as 1 - or initialize as an average of current approx
                # self.cce[key][0][a] = 1.0
                self.cce[key][1][a] = np.mean(self.cce[key][0])
        #--------------------------------



    def update_eq(self, state, action, loss,  eta=0.001, gamma=0.001, zeta = 0.9):   # normally eta = 0.1 but to help with small values
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
        s = tuple(state)                            # to enable mutability of dict keys
        A = max(self.action_space) + 1              # to the highest number of actions
        a = action                                  # an integer representing the action
        
        eq_approx, visit_count = {}, {}

        # update cce architecture if necessary
        if s not in self.cce or a not in self.cce[s][0]:
            self.initialize_or_update_cce(state, action)
        

        # loss passed in as tensor with grad after initial iteration typically
        if isinstance(loss, torch.Tensor):
            loss = loss.detach().numpy() if loss.requires_grad else loss.numpy()
        loss = np.sum(loss)


        # policy & action probabilities
        [eq_approx[s], visit_count[s]] = self.cce[s] 
        policy_t = eq_approx[s] / np.sum(eq_approx[s])
        # self.runningPolicy[s] += policy_t
        if np.sum(policy_t) == 0:
            print("Policy is zero for all actions")
            t = 5/0
        action_prob = policy_t[a] + 1e-50

        # estimate loss
        estimated_loss = loss / (action_prob + gamma)

        #--------------------- Trying due to vanishing equilibrium approximations
        # probabilities in log space to avoid numerical underflow
        log_probs = np.zeros_like(policy_t)
        log_probs = np.log(policy_t + 1e-50)
        log_probs[a] -= eta * estimated_loss
        max_log_prob = np.max(log_probs)
        #---------------------

        eq_approx[s][a] = np.exp(log_probs[a] - max_log_prob) # closer to the max prob the closer to 1
        visit_count[s][a] = visit_count[s][a] + 1
        # print("sum of current approx: ", np.sum(current_approx[s]))
        # current_approx[s] /= np.sum(current_approx[s])

        self.cce[s] = [eq_approx[s], visit_count[s]]  # Update the tuple in the dictionary

        # print("Action: ", a, "State: ", s, "Loss: ", loss, "Estimated Loss: ", estimated_loss, "Policy: ", policy_t[a])
        # print("cce[s][0]: ", self.cce[s][0])
        # print("cce[s][1]: ", self.cce[s][1])
        # print("sum of cce[s][0]: ", np.sum(self.cce[s][0]))
        # print("sum of cce[s][1]: ", np.sum(self.cce[s][1]))

        if any(np.isnan(self.cce[s][0])) or any(np.isnan(self.cce[s][1])):
            print("Nan in eq approx or count")
            for a in range(len(self.cce[s][0])):
                if np.isnan(self.cce[s][0][a]):
                    print("Nan in eq approx for action: ", a)
                if np.isnan(self.cce[s][1][a]):
                    print("Nan in count for action: ", a)
            if np.isnan(estimated_loss) or np.isnan(loss) or np.isnan(log_probs):
                print("nan in estimated loss, loss, or log probs")
            # print("estimated loss: ", estimated_loss, "loss: ", loss, "log prob: ", log_probs)
            t = 5 / 0 # breaks the loop and training run and prints the above messages

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
        print(current_approx[action])
        # print("Action: ", action)
        # print("Current count: ", current_count[action])
        # for i in range(len(current_count)):
        #     if current_count[i] > 0:
                # print(f"Action {i}: {current_count[i]}")
        # print("action: ", action, "count: ", current_count[action])
        # t = 5 / 0
        return action, current_count[action]
    
    def load_eq(self, path):
        '''
            Load the eq approximations from a file
        '''
        # self.eq_approx = torch.load(path)
        # return self.eq_approx
        # dict too large for torch.load -- use numpy instead
        with open(path, 'rb') as f:
            self.cce = pickle.load(f)
        # return self.cce
    
    




'''
If a bunch of running information is still printing out, go to :
cage-challege-2-main/CybORG/CybORG/Simulator/State.py
and comment out the print statements that are uncommented 

-- updates to this file are not being pushed to the main branch
'''
import numpy as np
import torch
import pickle

class CCE():
    def __init__(self):
        self.cce = {}  # Dictionary to store state to (action to [eq_approx, visit_count])
        
        self.gamma = 0.01
        self.eta = self.gamma * 2

    def initialize_or_update_cce(self, state, action):
        s = tuple(state)
        if s not in self.cce:
            self.cce[s] = {}

        if action not in self.cce[s]:
            # Check if there are any actions already for this state
            if self.cce[s]:  # If there are existing actions
                # Initialize as the average approx for all actions for this state, and visit count as 0
                avg_approx = np.mean([self.cce[s][a][0] for a in self.cce[s]])
            else:
                # If no actions exist, initialize to 1.0
                avg_approx = 1.0
            self.cce[s][action] = [avg_approx, 0]

    def update_eq(self, state, action, loss, T=10000):

        s = tuple(state)
        a = str(action)
        self.initialize_or_update_cce(s, a)

        if isinstance(loss, torch.Tensor):
            loss = loss.detach().numpy() if loss.requires_grad else loss.numpy()
        loss = np.sum(loss)

        eq_approx_dict = {act: val[0] for act, val in self.cce[s].items()}
        visit_count_dict = {act: val[1] for act, val in self.cce[s].items()}

        # Calculate the total sum of eq_approxs to find the policy
        total_eq_approx = sum(eq_approx_dict.values())
        policy = {act: eq_approx_dict[act] / total_eq_approx for act in eq_approx_dict}

        # Hyperparamter updates:
        # if len(policy) > 1:
        #     self.gamma = np.sqrt(2 * np.log(len(policy)) / (len(policy) * T))
        #     self.eta = 2 * self.gamma
        self.gamma = np.sqrt(2 * np.log(len(policy)) / (len(policy) * T))
        self.eta = 2 * self.gamma

        # print(f"Gamma: {self.gamma}, Eta: {self.eta}")

        estimated_loss = loss / (policy[a] + self.gamma)

        # Update the log probabilities
        log_probs = {act: np.log(policy[act] + 1e-50) for act in policy}
        log_probs[a] -= self.eta * estimated_loss

        # Normalize and exponentiate log probabilities
        max_log_prob = max(log_probs.values())
        for act in eq_approx_dict:
            eq_approx_dict[act] = np.exp(log_probs[act] - max_log_prob)
            self.cce[s][act][0] = eq_approx_dict[act]

        visit_count_dict[a] += 1
        self.cce[s][a][1] = visit_count_dict[a]
        
        return self.cce

    def get_all_eq_approximations(self, state):
        s = tuple(state)
        if s in self.cce:
            return {action: data[0] for action, data in self.cce[s].items()}
        else:
            print(f"No data available for state {s}.")
            return {}



    def save(self, path):
        '''
            Save the eq approximations to a file with pickle
        '''
        # saving cce with pickle since torch.load is not working for such a large file
        with open(path, 'wb') as f:
            pickle.dump(self.cce, f)

    
    def get_eq_action_visits(self, state, balance=100):
        '''
        Get the action based on the eq approximations.
        
        Args:
            state (list): The current state observation.

        Returns:
            tuple: The action with the highest eq approximation and its visit count.
        '''
        state_key = tuple(state)
        eq_approx_dict, visit_count_dict = {}, {}
        if state_key in self.cce:
            for a in self.cce[state_key]:
                eq_approx_dict[a] = self.cce[state_key][a][0]
                visit_count_dict[a] = self.cce[state_key][a][1]
            if not eq_approx_dict:
                return None, False  # or some default if no actions have been recorded yet

            '''
            # Find the action with the highest eq approximation
            best_action = max(eq_approx_dict, key=eq_approx_dict.get)
            # check if more than one action is the highest
            if list(eq_approx_dict.values()).count(eq_approx_dict[best_action]) > 1:
                print(f"Number of 'best actions' : {list(eq_approx_dict.values()).count(eq_approx_dict[best_action])}")
                # if so, choose the action with the highest visit count
                best_action = max(visit_count_dict, key=visit_count_dict.get)
            '''
            # For visits more than the balance point, choose the action with the highest eq approximation
            valid_actions = {a: eq_approx_dict[a] for a in eq_approx_dict if visit_count_dict[a] >= balance}
            if valid_actions:
                if list(valid_actions.values()).count(max(valid_actions.values())) > 1:
                    print(f"Multiple 'best actions' : {list(valid_actions.values()).count(max(valid_actions.values()))}")
                    best_actions = [a for a in valid_actions if valid_actions[a] == max(valid_actions.values())]
                    '''
                    Ok, two options, either chose the highest count out of the best options. OR:
                    select randomly out of the contenders
                    '''
                    best_action = max(best_actions, key=visit_count_dict.get)
                    #or
                    # best_action = np.random.choice(best_actions)

                best_action = max(valid_actions, key=valid_actions.get)
            else:
                print(f"No actions visited more than {balance} times.")
                return None, False  # or some default action, depending on requirements
            
            # best_eq_approx = eq_approx_dict[best_action]
            # visits = visit_count_dict[best_action]
            # print(best_action, best_eq_approx, visits)
            return best_action, True
        else:
            print(f"No data available for state {state_key}.")
            return None, False  # or some default action, depending on requirements
    
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
    
    
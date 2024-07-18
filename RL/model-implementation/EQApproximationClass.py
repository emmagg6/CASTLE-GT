import numpy as np

class EqApproximation:
    """
        EQ Approximation class that implements the EXP3-IX algorithm
    """
    # load states instaed of initialize it 
    # future optimize the code by using a dictionary to store the policy and visit count
    def __init__(self, states, num_actions, T):
        self.eq_approx = {}                     # dictionary tracks a str-state to a specific equilibrium approximation
        self.visit_count = {}                   # Maps state to a visit count np.array
        self.sumOfPolicy = {}                   # sum of the policy for all actions

        self.eq_approx_unknown = {}
        self.visit_count_unknown = {}

        for state in states:
            self.eq_approx[state] = np.full(num_actions, 1.0 / num_actions)                                         # initialize the policy to be uniform
            self.visit_count[state] = np.zeros(num_actions)                                                        # initialize the visit count to be zero
            self.sumOfPolicy[state] = np.zeros(num_actions)  # sum of the policy for all actions

        # for EXP3-IX algoritm
        self.total_time = T
        self.gamma = np.sqrt((2 * np.log(num_actions))/(num_actions + self.total_time)) 
        self.gamma_adapting = self.gamma
        self.eta = self.gamma * 2
        self.eta_adapting = self.eta
        self.num_actions = num_actions          # Number of actions

    def get_action(self, state):
        """
            get an action base on the policy and current state    
        """
        currentPolicy = self.eq_approx[state] / np.sum(self.eq_approx[state])                     # normalize the policy to a probability distribution
        action = np.random.choice(self.num_actions, p=currentPolicy)                    # choose an action using the policy 
        self.visit_count[state][action] += 1                                                    # increment the visit count for the chosen action
        return action
    
    def get_action_adaptation(self, state, zeta = 1):
        """
            extension for larger environments
        """
        state = str(state)
        # check is state is in the eq_approx_unknown dictionary
        if state not in self.eq_approx_unknown:
            return None
        # get the action with the highest value, with visit counts at least as many as zeta value
        valid_action = []
        for action in self.eq_approx_unknown[state]:
            if self.visit_count_unknown[state][action] >= zeta:
                valid_action.append(action)
        # get the action with the highest value out for all of the actions in the valid_action list
        action = max(valid_action, key=lambda x: self.eq_approx_unknown[state][x])
        # print(f"CCE Action: {action}")
        return action

    def update_policy(self, state, chosen_action, loss):
        """
            update the policy for the respective state and chosed action based on the loss
        """
        
        curerntPolicy = self.eq_approx[state] / np.sum(self.eq_approx[state])                                                # normalize the policy to a probability distribution
        self.sumOfPolicy[state] += curerntPolicy
        action_prob = curerntPolicy[chosen_action]

        estimated_loss = loss / (action_prob + self.gamma)                                                             # estimated loss for the chosen action
        self.eq_approx[state][chosen_action] *= np.exp(-self.eta * estimated_loss)                              # update the policy for the chosen action


    # def observe_and_update(self, state, action, loss):
    #     """
    #         used when the agent is not the one interacting with the environment
    #     """
        
    #     self.visit_count[state][action] += 1
        
    #     currentPolicy = self.eq_approx[state] / np.sum(self.eq_approx[state])                                    # normalize the policy to a probability distribution
    #     self.sumOfPolicy[state] += currentPolicy                                                                 # sum of the policy for all actions

    #     action_prob = currentPolicy[action]

    #     estimated_loss = loss / ((action_prob+ 1e-50) + self.gamma)

    #     self.eq_approx[state][action] *= np.exp(-self.eta * estimated_loss)
  
        
    def observe_and_update_adaptation(self, state, action, loss):
        """
            extension for larger environments
        """
        state = str(state)

        # Initialize state and action in visit_count_unknown if not already present
        if state not in self.visit_count_unknown:
            self.visit_count_unknown[state] = {}
        if action not in self.visit_count_unknown[state]:
            self.visit_count_unknown[state][action] = 0

        # Increment the visit count
        self.visit_count_unknown[state][action] += 1

        # Ensure the eq_approx_unknown dictionary is properly initialized
        if state not in self.eq_approx_unknown:
            self.eq_approx_unknown[state] = {}
        if action not in self.eq_approx_unknown[state]:
            self.eq_approx_unknown[state][action] = 1.0  # Initial value for unknown action
        else:
            # Initialize action value as the mean of existing action values for the state
            self.eq_approx_unknown[state][action] = np.mean(list(self.eq_approx_unknown[state].values()))

        total_sum = np.sum(list(self.eq_approx_unknown[state].values()))
        currentPolicy = {act: val / total_sum for act, val in self.eq_approx_unknown[state].items()}
                                                           # sum of the policy for all actions
        
        ##### update hyperparameters #####
        self.gamma_adapting = np.sqrt((2 * np.log(len(currentPolicy)))/(len(currentPolicy) + self.total_time))
        self.eta_adapting = self.gamma * 2
        ##################################

        action_prob = currentPolicy[action]

        estimated_loss = loss / ((action_prob + 1e-50) + self.gamma)

        log_probs = {act: np.log(currentPolicy[act] + 1e-50) for act in currentPolicy}
        log_probs[action] -= self.eta * estimated_loss

        max_log_prob = max(log_probs.values())
        for act in currentPolicy:
            self.eq_approx_unknown[state][act] = np.exp(log_probs[act] - max_log_prob)

    
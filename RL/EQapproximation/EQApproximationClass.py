import numpy as np

class EqApproximation:
    """
        EQ Approximation class that implements the EXP3-IX algorithm
    """
    # load states instaed of initialize it 
    # future optimize the code by using a dictionary to store the policy and visit count
    def __init__(self, states, num_actions, eta, gamma):
        self.eq_approx = {}                     # dictionary tracks a str-state to a specific equilibrium approximation
        self.visit_count = {}                   # Maps state to a visit count np.array
        self.sumOfPolicy = {}                   # sum of the policy for all actions

        self.eq_approx_unknown = {}

        for state in states:
            self.eq_approx[state] = np.full(num_actions, 1.0 / num_actions)                                         # initialize the policy to be uniform
            self.visit_count[state] = np.zeros(num_actions)                                                        # initialize the visit count to be zero
            self.sumOfPolicy[state] = np.zeros(num_actions)  # sum of the policy for all actions

        # for EXP3-IX algoritm
        self.eta = eta                          # Learning rate
        self.gamma = gamma                      # Exploration parameter
        self.num_actions = num_actions          # Number of actions

    def get_action(self, state):
        """
            get an action base on the policy and current state    
        """
        currentPolicy = self.eq_approx[state] / np.sum(self.eq_approx[state])                     # normalize the policy to a probability distribution
        action = np.random.choice(self.num_actions, p=currentPolicy)                    # choose an action using the policy 
        self.visit_count[state][action] += 1                                                    # increment the visit count for the chosen action
        return action

    def update_policy(self, state, chosen_action,loss):
        """
            update the policy for the respective state and chosed action based on the loss
        """
        
        curerntPolicy = self.eq_approx[state] / np.sum(self.eq_approx[state])                                                # normalize the policy to a probability distribution
        self.sumOfPolicy[state] += curerntPolicy
        action = curerntPolicy[chosen_action]

        estimated_loss = loss / (action + self.gamma)                                                             # estimated loss for the chosen action
        self.eq_approx[state][chosen_action] *= np.exp(-self.eta * estimated_loss)                              # update the policy for the chosen action


    def observe_and_update(self, state, action, loss):
        """
            used when the agent is not the one interacting with the environment
        """
        
        self.visit_count[state][action] += 1
        
        currentPolicy = self.eq_approx[state] / np.sum(self.eq_approx[state])                                    # normalize the policy to a probability distribution
        self.sumOfPolicy[state] += currentPolicy                                                                 # sum of the policy for all actions

        action_prob = currentPolicy[action]

        estimated_loss = loss / ((action_prob+ 1e-50) + self.gamma)

        self.eq_approx[state][action] *= np.exp(-self.eta * estimated_loss)
  
        
    def observe_and_update_adaptation(self, state, action, loss):
        """
            extension for larger environments
        """
        
        self.visit_count[state][action] += 1

        if state not in self.eq_approx_unknown: # if state is unknown, then no actions are here so initialize as 1
            self.eq_approx_unknown[state][action] = 1.0
            if action not in self.eq_approx_unknown[state]: # if new action then initialize as the mean of the current approx values for this state
                self.eq_approx_unknown[state][action] = np.mean([self.eq_approx_unknown[state][act] for act in self.eq_approx_unknown[state]])
             
        
        currentPolicy = self.eq_approx_unknown[state] / np.sum(self.eq_approx_unknown[state])                                    # normalize the policy to a probability distribution
        self.sumOfPolicy[state] += currentPolicy                                                                 # sum of the policy for all actions

        action_prob = currentPolicy[action]

        estimated_loss = loss / ((action_prob+ 1e-50) + self.gamma)

        log_probs = {act: np.log(currentPolicy[act] + 1e-50) for act in currentPolicy}
        log_probs[a] -= self.eta * estimated_loss

        max_log_prob = max(log_probs.values())
        for act in currentPolicy:
            self.eq_approx_unknown[state][act] = np.exp(log_probs[act] - max_log_prob)

        # self.eq_approx[state][action] = np.exp(-self.eta * estimated_loss)
    
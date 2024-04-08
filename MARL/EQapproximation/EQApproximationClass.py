import numpy as np

class EqApproximation:
    """
        EQ Approximation class that implements the EXP3-IX algorithm
    """
    def __init__(self, states, num_actions, eta, gamma):
        self.eq_approx = {}                     # dictionary tracks a str-state to a specific equilibrium approximation
        self.visit_count = {}                   # Maps state to a visit count np.array

        for state in states:
            self.eq_approx[state] = np.full(num_actions, 1.0 / num_actions)                # initialize the policy to be uniform
            self.visit_count[state] = np.zeros(num_actions)                                # initialize the visit count to be zero

        # for EXP3-IX algoritm
        self.eta = eta                          # Learning rate
        self.gamma = gamma                      # Exploration parameter
        self.num_actions = num_actions          # Number of actions

    def get_action(self, state):
        """
            get an action base on the policy and current state    
        """
        action = np.random.choice(self.num_actions, p=self.eq_approx[state])                    # choose an action using the policy 
        self.visit_count[state][action] += 1                                                    # increment the visit count for the chosen action
        return action

    def update_policy(self, state, chosen_action,loss):
        """
            update the policy for the respective state and chosed action based on the loss
        """
        estimated_loss = loss / self.eq_approx[state][chosen_action] + self.gamma                               # estimated loss for the chosen action
        self.eq_approx[state][chosen_action] *= np.exp(-self.eta * estimated_loss)                              # update the policy for the chosen action
        self.eq_approx[state] /= np.sum(self.eq_approx[state])                                                  # normalize the policy to a probability distribution

    def observe_and_update(self, state, action, loss):
        """
            used when the agent is not the one interacting with the environment
        """
        self.visit_count[state][action] += 1
        estimated_loss = loss / self.eq_approx[state][action] + self.gamma
        self.eq_approx[state][action] *= np.exp(-self.eta * estimated_loss)
        self.eq_approx[state] /= np.sum(self.eq_approx[state])  
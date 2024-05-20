import numpy as np

class QLearningAgent:
    """
        implemented base on Q learning algorithm
    """
    def __init__(self, num_states, num_actions, alpha, gamma,epsilon):
        self.q_table = np.zeros((num_states, num_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.num_actions = num_actions
        self.epsilon = epsilon

    def choose_action(self, state):
        """ 
            choose an action base on the epsilon-greedy policy
            for now: we dont explore, only exploit
        """
        if np.random.random() < self.epsilon:
            return np.random.choice(self.num_actions)  
        else:
            return np.argmax(self.q_table[state]) 
        
    def update_q_table(self, state, action, loss, next_state):
        """
            use this because the environment is to minimize the loss instead of maximizing the reward
        """
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = -loss + self.gamma * self.q_table[next_state][best_next_action]  
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error
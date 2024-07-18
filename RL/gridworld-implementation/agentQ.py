import numpy as np

class QLearningAgent:
    def __init__(self, q_table = np.zeros((1, 1, 1)), size = 0, actions = [], alpha=0.1, gamma=0.9, epsilon=0.25):
        self.size = size
        self.actions = actions  # list of possible actions
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.q_table = q_table

    def get_action(self, state):
        # epsilon-greedy policy
        if np.random.uniform(0, 1) < self.epsilon:
            # explore: choose a random action
            action_index = np.random.choice(len(self.actions))
        else:
            # exploit: choose the action with max Q-value for the current state
            action_index = np.argmax(self.q_table[state[0], state[1]])
        return self.actions[action_index], action_index

    def update_q_table(self, state, action_index, reward, next_state):
        old_value = self.q_table[state[0], state[1], action_index]
        next_max = np.max(self.q_table[next_state[0], next_state[1]])

        # Q-learning update rule
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[state[0], state[1], action_index] = new_value
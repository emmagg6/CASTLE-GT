import numpy as np
class RedAgent:
    """
        a random agent that can change the state of the environment
    """
    def __init__(self):
        self.actions = []

    def get_action(self, actions):
        return np.random.choice(actions)

    
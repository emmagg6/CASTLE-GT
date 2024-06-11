import numpy as np

class Stochasticity:
    """
        randomness that can change the state of the environment
    """
    def __init__(self):
        self.actions = []

    def get_action(self, actions):
        return np.random.choice(actions)

    
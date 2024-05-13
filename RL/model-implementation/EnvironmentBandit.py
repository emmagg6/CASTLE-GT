class Environment:
    """
    A deterministic simulated environment for a multi-armed bandit problem where agents interact with different actions.
    
    States:
    - There are 11 states, ranging from 0 (no host infected) to 10 (more than 10 hosts infected).
    
    Actions:
    - Actions are generically named to fit the bandit problem theme, e.g., "Action 1", "Action 2", etc.
    
    The environment deterministically adjusts based on the selected actions.
    """
    def __init__(self):
        self.states = list(range(11))  # states from 0 to 10
        self.current_state = 0
        self.losses = [0, 0.1, 0.12, 0.14, 0.16, 0.20, 0.25, 0.30, 0.40, 0.50, 1]

    def step(self, blueAction, redAction):
        """
        Process actions from blue and red agents, update the state deterministically, and return the current state and associated loss.
        """
        # Red agent's impact
        if redAction == "Action 1":
            self.current_state -= 1  # Deterministic increase for red action (opponent of blue)
        else:
            self.current_state += 0  # No change for red action

        # Blue agent's impact
        if blueAction == "Action 2":
            self.current_state -= 2  # Deterministic decrease for blue action
        elif blueAction == "Action 3":
            self.current_state += 1  # Deterministic slight increase for blue action
        elif blueAction == "Action 4":
            self.current_state += 2  # Deterministic greater increase for blue action

        # Ensure state boundaries are respected
        self.current_state = max(0, min(self.current_state, 10))
        
        # Determine the loss from the current state
        current_loss = self.losses[self.current_state]

        return self.current_state, current_loss

    def get_loss(self, blueAction):
        """
        get the loss for the respective state; used for calculating the regret
        """
        state = self.current_state

         # Blue agent's impact
        if blueAction == "Action 2":
            state -= 2  # Deterministic decrease for blue action
        elif blueAction == "Action 3":
            state += 1  # Deterministic slight increase for blue action
        elif blueAction == "Action 4":
            state += 2  # Deterministic greater increase for blue action

        # Ensure state boundaries are respected
        state = max(0, min(state, 10))
        
        # Determine the loss from the current state
        current_loss = self.losses[state]

        return current_loss
 
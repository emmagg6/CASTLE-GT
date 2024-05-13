class Environment:
    """
        an demo environment for the blue agent
        this environment is intended by the blue agent and red agents to interact with the environment 

        states: 11 states; 0 state is no attack; 1 state is 1 host gets attacked, etc
    """
    def __init__(self):
        
        """
        for now, the states and loss are just simple values for demonstration

        states: there are 11 states in total; state 0 means no host are infected; state 2 means 2 hosts are infected;state 10 means more than 10 are infected

        """
        self.states = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]        # 0: no infection, 1: 1 infected, 2: 2 infected, 3: 3 infected, 4: 4 infected
        self.current_state = 0
        self.losses = {                                         # key: state; value: loss for each action; loss has to be congfigured between 0 and 1
            0: 0,                                               # if no infected                    
            1: 0.1,                                              # if 1 infected            
            2: 0.12,                                              # if 2 infected
            3: 0.14,
            4: 0.16,
            5: 0.20,
            6: 0.25,
            7: 0.30,
            8: 0.40,
            9: 0.50,
            10: 1                                             # more than 10 cotamination
        }

    def step(self, blueAction, redAction):
        """
        demo function: take the action of the blue agent and the red agent and return the next state and the loss
        """
        if blueAction == "Remove":                              
            self.current_state -= 4
        if blueAction == "Sleep" or blueAction == "Restore":
            self.current_state += 1
        if redAction == "Attack":
            self.current_state += 2

        if self.current_state > len(self.states)-1:             # if more than 10 contaminated -> stay at state 10
            self.current_state = len(self.states)-1

        if self.current_state < 0:                              # if less than 0 contaminated -> stay at state 0
            self.current_state = 0
            
        return self.current_state, self.losses[self.current_state]
 
    def get_loss(self, blueAction):
        """
        get the loss for the respective state; used for calculating the regret
        """
        state = self.current_state
        if blueAction == "Remove":
            state -= 4
        if blueAction == "Sleep" or blueAction == "Restore":
            state += 1

        if state > len(self.states)-1:             # if more than 10 contaminated -> stay at state 10
            state = len(self.states)-1

        if state < 0:                              # if less than 0 contaminated -> stay at state 0
            state = 0
            
        return self.losses[state]
 
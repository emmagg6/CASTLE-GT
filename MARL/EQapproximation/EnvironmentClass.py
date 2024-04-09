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
            2: 0.2,                                              # if 2 infected
            3: 0.4,
            4: 0.6,
            5: 0.8,
            6: 0.85,
            7: 0.9,
            8: 0.95,
            9: 0.96,
            10: 1                                             # more than 10 cotamination
        }

    def step(self, blueAction, redAction):
        """
        demo function: take the action of the blue agent and the red agent and return the next state and the loss
        """
        if blueAction == "Remove":
            self.current_state -= 1
        if redAction == "attack":
            self.current_state += 1

        if self.current_state > len(self.states)-1:             # if more than 10 contaminated -> stay at state 10
            self.current_state = len(self.states)-1

        if self.current_state < 0:                              # if less than 0 contaminated -> stay at state 0
            self.current_state = 0
            
        return self.current_state, self.losses[self.current_state]
    


    


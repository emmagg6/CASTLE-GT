class RedAgent:
    """
        a demo agent that infected 1 host after a fixed interval
    """
    def __init__(self, interval):
        self.interval = interval
        self.current_tick = 0

    def get_action(self):
        self.current_tick += 1
        if self.current_tick % self.interval == 0:
            return "Action 1"                           # attack
        return "Action 0"                               # noOp
    

    
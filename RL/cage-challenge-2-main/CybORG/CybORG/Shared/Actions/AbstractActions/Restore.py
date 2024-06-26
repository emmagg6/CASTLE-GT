from random import choice

from CybORG.Shared import Observation
from .Monitor import Monitor
from CybORG.Shared.Actions import Action
from CybORG.Shared.Actions.ConcreteActions.RestoreFromBackup import RestoreFromBackup
from CybORG.Simulator.Session import VelociraptorServer
from CybORG.Shared.Actions.AbstractActions import Monitor
from CybORG.Operations.Cost import Cost

class Restore(Action):
    def __init__(self, session: int, agent: str, hostname: str):
        super().__init__()
        self.agent = agent
        self.session = session
        self.hostname = hostname

    def sim_execute(self, state) -> Observation:
        # perform monitor at start of action
        #monitor = Monitor(session=self.session, agent=self.agent)
        #obs = monitor.sim_execute(state)

        if self.session not in state.sessions[self.agent]:
            return Observation(False)
        parent_session: VelociraptorServer = state.sessions[self.agent][self.session]
        # find relevant session on the chosen host
        sessions = [s for s in state.sessions[self.agent].values() if s.host == self.hostname]
        if len(sessions) > 0:
            session = choice(sessions)
            obs = Observation(True)
            # restore host
            action = RestoreFromBackup(session=self.session, agent=self.agent, target_session=session.ident)
            action.sim_execute(state)
            # remove suspicious files
            return obs
        else:
            return Observation(False)

    def operations_cost(self, scenario) -> int:
        subnet_names = scenario.get_host_subnet_names(self.hostname)
        
        # if host is present in multiple subnets, select subnet with highest penalty
       
        cost = Cost() 
        cost_per_subnet = cost.cost_per_subnet

        opcost = 0 # some large number; cost is negative

        for subnet in subnet_names:
            cost_dict = cost_per_subnet[subnet]
            c = cost_dict[self.hostname]
            if c < opcost:
                opcost = c

        return opcost

    #@property
    #def cost(self):
    #    return -1

    def __str__(self):
        return f"{self.__class__.__name__} {self.hostname}"

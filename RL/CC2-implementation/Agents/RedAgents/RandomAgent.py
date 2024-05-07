from random import choice
from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent
from CybORG.Shared.Actions import Sleep,DiscoverRemoteSystems,DiscoverNetworkServices,ExploitRemoteService,BlueKeep,EternalBlue,FTPDirectoryTraversal,HarakaRCE,HTTPRFI,HTTPSRFI,SQLInjection,PrivilegeEscalate,Impact,SSHBruteForce


class RandomAgent(BaseAgent):
    
    def train(self, results):
        pass

    def flatten_action_space(self,action_space):
        # Go through all action space and enumerate every possible action so we can choose one at random
        possible_actions = []
        for action in filter(action_space['action'].get, action_space['action']):
            if action == Sleep:
                possible_actions.append(action())
            elif action == DiscoverRemoteSystems:
                [possible_actions.append(action(session=session,agent=agent,subnet=subnet))
                            for session in filter(action_space['session'].get, action_space['session']) 
                            for agent in filter(action_space['agent'].get, action_space['agent'])
                            for subnet in filter(action_space['subnet'].get, action_space['subnet'])]
            elif action == DiscoverNetworkServices or action == ExploitRemoteService:
                [possible_actions.append(action(session=session,agent=agent,ip_address=ip_address))
                            for session in filter(action_space['session'].get, action_space['session']) 
                            for agent in filter(action_space['agent'].get, action_space['agent'])
                            for ip_address in filter(action_space['ip_address'].get, action_space['ip_address'])]
            elif action == BlueKeep or action == EternalBlue or action == FTPDirectoryTraversal or action == HarakaRCE or action == HTTPRFI or action == HTTPSRFI or action == SQLInjection or action == SSHBruteForce:
                [possible_actions.append(action(session=session,agent=agent,ip_address=ip_address,target_session=target_session))
                            for session in filter(action_space['session'].get, action_space['session'])
                            for agent in filter(action_space['agent'].get, action_space['agent'])
                            for ip_address in filter(action_space['ip_address'].get, action_space['ip_address'])
                            for target_session in filter(action_space['target_session'].get, action_space['target_session'])]
            elif action == Impact or action == PrivilegeEscalate:
                [possible_actions.append(action(session=session,agent=agent,hostname=hostname))
                            for session in filter(action_space['session'].get, action_space['session']) 
                            for agent in filter(action_space['agent'].get, action_space['agent'])
                            for hostname in filter(action_space['hostname'].get, action_space['hostname'])]
            else:
                raise NotImplementedError()
        return possible_actions

    def take_random_action(self, action_space):
        action_space_array = self.flatten_action_space(action_space)
        action = choice(action_space_array)
        return action

    def get_action(self, observation, action_space):
        return self.take_random_action(action_space)

    def end_episode(self):
        pass

    def set_initial_values(self, action_space, observation):
        pass

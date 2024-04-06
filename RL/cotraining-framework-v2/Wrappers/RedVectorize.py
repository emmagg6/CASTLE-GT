import math
from copy import deepcopy
from prettytable import PrettyTable
import numpy as np
from pprint import pprint
from CybORG.Shared.Actions.ConcreteActions.ExploitAction import ExploitAction

class RedVectorize:
    def __init__(self):
        self.red_info = {}
        self.known_subnets = set()
        self.step_counter = -1
        self.id_tracker = -1
        self.success = None

    def reset(self, obs):
        print("Reseting Red Vectorize")
        #self._process_initial_obs(obs)
        self.red_info = {}
        self.known_subnets = set()
        self.step_counter = -1
        self.id_tracker = -1
        self.success = None

        obs = self.observation_change(obs)
        return obs

    def observation_change(self, observation, action):
        self.success = observation['success']

        self.step_counter += 1
        if self.step_counter <= 0:
            self._process_initial_obs(observation)
        elif self.success:
        #elif self.success.__eq__(True):
            #print("Success val: ", self.success)
            self._update_red_info(observation, action)

        obs = self._create_vector()

        return obs


    def _process_initial_obs(self, obs):
        # print("Processing initial observation")
        for hostid in obs:
            if hostid == 'success':
                continue
            host = obs[hostid]
            interface = host['Interface'][0]
            subnet = interface['Subnet']
            self.known_subnets.add(subnet)
            ip = str(interface['IP Address'])
            hostname = host['System info']['Hostname']
            self.red_info[ip] = [str(subnet), str(ip), hostname, False, 'Privileged']
        
        #print(self.red_info)
   

    def _update_red_info(self, obs, action):
        #action = self.get_last_action(agent='Red')
        name = action.__class__.__name__
        if name == 'DiscoverRemoteSystems':
            self._add_ips(obs)
        elif name == 'DiscoverNetworkServices':
            if len(obs) > 1:
                tmp = obs.popitem()
                ip = str(tmp[1]['Interface'][0]['IP Address'])
                self.red_info[ip][3] = True
        elif name == 'ExploitRemoteService' or isinstance(action, ExploitAction):
            self._process_exploit(obs)
        elif name == 'PrivilegeEscalate':
            hostname = action.hostname
            try:
                self._process_priv_esc(obs, hostname)
            except IndexError as e:
                print("Priv esc error:", self.red_info)
        #elif name != 'InvalidAction':
            # raise ValueError('Action of incorrect type for RedTableWrapper, infos have not been taken into account')


    def _generate_name(self, datatype: str):
        self.id_tracker += 1
        unique_id = 'UNKNOWN_' + datatype + ': ' + str(self.id_tracker)
        return unique_id

    def _add_ips(self, obs):
        for hostid in obs:
            if hostid == 'success':
                continue
            host = obs[hostid]
            for interface in host['Interface']:
                ip = interface['IP Address']
                subnet = interface['Subnet']
                if subnet not in self.known_subnets:
                    self.known_subnets.add(subnet)
                if str(ip) not in self.red_info:
                    subnet = self._get_subnet(ip)
                    hostname = self._generate_name('HOST')
                    self.red_info[str(ip)] = [subnet, str(ip), hostname, False, 'None']
                elif self.red_info[str(ip)][0].startswith('UNKNOWN_'):
                    self.red_info[str(ip)][0] = self._get_subnet(ip)

    def _get_subnet(self, ip):
        for subnet in self.known_subnets:
            if ip in subnet:
                return str(subnet)
        return self._generate_name('SUBNET')

    def _process_exploit(self, obs):
        for hostid in obs:
            if hostid == 'success':
                continue

            host = obs[hostid]
            if 'Sessions' in host:
                ip = str(host['Interface'][0]['IP Address'])
                hostname = host['System info']['Hostname']
                session = host['Sessions'][0]
                access = 'Privileged' if 'Username' in session else 'User'

                self.red_info[ip][2] = hostname
                self.red_info[ip][4] = access

    def _process_priv_esc(self, obs, hostname):
        if obs['success'] == False:
            [info for info in self.red_info.values() if info[2] == hostname][0][4] = 'None'
        else:
            for hostid in obs:
                if hostid == 'success':
                    continue
                host = obs[hostid]
                ip = host['Interface'][0]['IP Address']
    
                if 'Sessions' in host:
                    access = 'Privileged'
                    self.red_info[str(ip)][4] = access
                else:
                    subnet = self._get_subnet(ip)
                    hostname = self._generate_name('HOST')
    
                    if str(ip) not in self.red_info:
                        self.red_info[str(ip)] = [subnet, str(ip), hostname, False, 'None']
                    else:
                        self.red_info[str(ip)][0] = subnet
                        self.red_info[str(ip)][2] = hostname


    def _create_red_table(self):
        # The table data is all stored inside the ip nodes
        # which form the rows of the table
        table = PrettyTable([
            'Subnet',
            'IP Address',
            'Hostname',
            'Scanned',
            'Access',
        ])
        for ip in self.red_info:
            table.add_row(self.red_info[ip])

        table.sortby = 'IP Address'
        table.success = self.success

        #pprint(table)
        return table

    def _create_vector(self, num_hosts=13):
        table = self._create_red_table()._rows

        # Compute required length of vector based on number of hosts
        padding = num_hosts - len(table)
        id_length = math.ceil(math.log2(num_hosts))

        success_value = int(self.success.value) if self.success.value < 2 else -1
        proto_vector = [success_value]
        for row in table:
            # Scanned
            proto_vector.append(int(row[3]))

            # Access
            access = row[4]
            if access == 'None':
                value = [0, 0]
            elif access == 'User':
                value = [1, 0]
            elif access == 'Privileged':
                value = [0, 1]
            else:
                raise ValueError('Table had invalid Access Level')
            proto_vector.extend(value)

        proto_vector.extend(padding * 3 * [-1])

        return np.array(proto_vector)


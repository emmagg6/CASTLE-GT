
class Cost:

    def __init__(self):
        # Enterprise 1 has high traffic, expensive Restore: cost(E1) = -8; Remove suspicious processes is preferred
        # Enterprise 2 connects to OpServer as root, Restore is imperative: cost(E2) = 0
        #self.cost_per_subnet = {"User": -0.1, "Enterprise": [-0.1, -8, 0, -0.1], "Operational": -10}

        '''
        self.cost_per_subnet = {"User": {'User0':-0.1, 'User1':-0.1, 'User2':-0.1, 'User3':-0.1, 'User4':-0.1}, 
                                "Enterprise": {'Enterprise0':-0.1, 'Enterprise1':-8, 'Enterprise2':0, 'Defender':-0.1}, 
                                "Operational": {'Op_Server0':-10, 'Op_Host0':-0.1, 'Op_Host1':-0.1, 'Op_Host2':-0.1}}
        '''
        ''' 
        self.cost_per_subnet = {"User": {'User0':-0.1, 'User1':-0.1, 'User2':-0.1, 'User3':-0.1, 'User4':-0.1}, 
                                "Enterprise": {'Enterprise0':-1, 'Enterprise1':-8, 'Enterprise2':0, 'Defender':-1}, 
                                "Operational": {'Op_Server0':-10, 'Op_Host0':-0.1, 'Op_Host1':-0.1, 'Op_Host2':-0.1}}
        '''
        ''' 
        self.cost_per_subnet = {"User": {'User0':-0.1, 'User1':-0.1, 'User2':-0.1, 'User3':-0.1, 'User4':-0.1}, 
                                "Enterprise": {'Enterprise0':-1.5, 'Enterprise1':-3, 'Enterprise2':0, 'Defender':-0.5}, 
                                "Operational": {'Op_Server0':-10, 'Op_Host0':-0.1, 'Op_Host1':-0.1, 'Op_Host2':-0.1}}
        ''' 
        '''
        self.cost_per_subnet = {"User": {'User0':-0.1, 'User1':-0.1, 'User2':-0.1, 'User3':-0.1, 'User4':-0.1}, 
                                "Enterprise": {'Enterprise0':-0.5, 'Enterprise1':-4, 'Enterprise2':0, 'Defender':-0.5}, 
                                "Operational": {'Op_Server0':-10, 'Op_Host0':-0.1, 'Op_Host1':-0.1, 'Op_Host2':-0.1}}
        
        '''
        '''
        self.cost_per_subnet = {"User": {'User0':-0.1, 'User1':-0.1, 'User2':-0.1, 'User3':-0.1, 'User4':-0.1}, 
                                "Enterprise": {'Enterprise0':-3, 'Enterprise1':-6, 'Enterprise2':0, 'Defender':-1}, 
                                "Operational": {'Op_Server0':-10, 'Op_Host0':-0.1, 'Op_Host1':-0.1, 'Op_Host2':-0.1}}
        '''
       
        '''
        self.cost_per_subnet = {"User": {'User0':-0.1, 'User1':-0.1, 'User2':-0.1, 'User3':-0.1, 'User4':-0.1}, 
                                "Enterprise": {'Enterprise0':-0.2, 'Enterprise1':-1.6, 'Enterprise2':0, 'Defender':-0.2}, 
                                "Operational": {'Op_Server0':-10, 'Op_Host0':-0.1, 'Op_Host1':-0.1, 'Op_Host2':-0.1}}
        '''
        
        ''' 
        self.cost_per_subnet = {"User": {'User0':-0.1, 'User1':-0.1, 'User2':-0.1, 'User3':-0.1, 'User4':-0.1}, 
                                "Enterprise": {'Enterprise0':-0.6, 'Enterprise1':-1.2, 'Enterprise2':0, 'Defender':-0.2}, 
                                "Operational": {'Op_Server0':-10, 'Op_Host0':-0.1, 'Op_Host1':-0.1, 'Op_Host2':-0.1}}
        '''

        ''' 
        self.cost_per_subnet = {"User": {'User0':0, 'User1':0, 'User2':0, 'User3':0, 'User4':0}, 
                                "Enterprise": {'Enterprise0':0, 'Enterprise1':0, 'Enterprise2':0, 'Defender':0}, 
                                "Operational": {'Op_Server0':0, 'Op_Host0':0, 'Op_Host1':0, 'Op_Host2':0}}
        '''

       
        '''
        # hops_u0_e16_02_02_e3
        self.cost_per_subnet = {"User": {'User0':0, 'User1':0, 'User2':0, 'User3':0, 'User4':0}, 
                                "Enterprise": {'Enterprise0':-1.6, 'Enterprise1':-0.2, 'Enterprise2':-3, 'Defender':-0.2}, 
                                "Operational": {'Op_Server0':-10, 'Op_Host0':0, 'Op_Host1':0, 'Op_Host2':0}}
        '''
        
        
        # hops_u01_e16_02_02_e3
        self.cost_per_subnet = {"User": {'User0':-0.1, 'User1':-0.1, 'User2':-0.1, 'User3':-0.1, 'User4':-0.1},
                                "Enterprise": {'Enterprise0':-1.6, 'Enterprise1':-0.2, 'Enterprise2':-3, 'Defender':-0.2},
                                "Operational": {'Op_Server0':-10, 'Op_Host0':0, 'Op_Host1':0, 'Op_Host2':0}}
        

from EnvironmentClass import Environment
from EQApproximationClass import EqApproximation
from ThirdAgentClass import QLearningAgent
from RedAgent import RedAgent

from utility import plot_graph
from utility import plot_scaled_sum_policy_over_time

def EQasAgent():
    """
        blueAgent (EQ approximator) will be the one interacting with the environment
    """
    print("start running EQasAgent")
    # Initialization 1: Environment
    env = Environment()
    states = env.states

    # Initialization 2: EQ Approximation                                                                 # Exploration parameter, γ
    blueAgentActions = ["Analyse","Remove"]                                      # for simplicity, the blue agent can only choose between two actions
    redAgentAction = ["noOp","attack"]
    blueAgent = EqApproximation(states, num_actions=2, eta=0.1, gamma=0.01)     # gamma and eta set to 0.01 and 0.1 for simplicity

    redAgent = RedAgent(5)                                                      # for simplicity: red agent will attack once for 5 time step 
    
    # for plotting and loggin purposes
    cum_loss = []
    log=[]

    for _ in range(100000): 
        state = env.current_state 
        blueActionIndex = blueAgent.get_action(state)
        RedActionIndex = redAgent.get_action()

        blueAction = blueAgentActions[blueActionIndex]
        RedAction = redAgentAction[RedActionIndex]

        nextstate,loss = env.step(blueAction,RedAction) 
        blueAgent.update_policy(state, blueActionIndex,loss)

        cum_loss.append(loss)
        log.append([blueAgentActions[blueActionIndex],state,loss])

    # Output the policy and visit counts
    print(f"Policy for state '{state}':", blueAgent.eq_approx[state])
    print(f"Visit counts for state '{state}':", blueAgent.visit_count[state])

    # print("log",log)
    plot_graph(cum_loss,name="EQasAgent")

def EQasObserver():
    """
        blueAgent (Q learning agent) will be the one interacting with the environment
        EQobserver will observe the interaction and update its policy

        => compare the EQ approximator policy with the Q learning agent policy
    """
    # Initialization 1: Environment
    env = Environment()
    states = env.states

    # Initialization 2: EQ Approximation
    num_states = len(env.states)                                                                                         
    blueAgentActions = ["Remove","Restore","sleep"]                                                      # actions for the blue agent in the cage 4
    EQobserver = EqApproximation(states, num_actions=3, eta=0.1, gamma=0.01)                                             # gamma: exploration; eta: learning rate
    blueAgent = QLearningAgent(num_states, num_actions=3, alpha=0.1, gamma=0.8, epsilon=0.01)                            # gamma: discounting; alpha: learning rate; epsilon: exploration

    # initialization 3:   
    redAgentAction = ["noOp","attack"]
    redAgent = RedAgent(5)      

    cum_loss = []
    log=[]
    policy_sums_over_time = {state: [] for state in EQobserver.sumOfPolicy.keys()}  # Prepare dictionary to store the data over time
    checkpoints = []

    total_iteration=1000
    for iteration in range(total_iteration): 
        state = env.current_state                                                   # get the current state from teh environment
        BlueActionIndex = blueAgent.choose_action(state)                                # let the Q learning agent choose an action
        
        RedActionIndex = redAgent.get_action()
        RedAction = redAgentAction[RedActionIndex]
        next_state, loss = env.step(BlueActionIndex,RedAction)                                    # get the loss for the chosen action
        blueAgent.update_q_table(state, BlueActionIndex,loss,next_state)                # update the policy for the Q learning agent

        # update for EQ Appriximation as observer
        EQobserver.observe_and_update(state, BlueActionIndex, loss)

        cum_loss.append(loss)
        log.append([blueAgentActions[BlueActionIndex],state,loss])

        # Print the sum of policy snapshot every 100 iterations
        if (iteration + 1) % 100 == 0:
            # print(f"Iteration {iteration + 1}: Sum of Policy Snapshot:")
            checkpoints.append(iteration + 1)  # Record the checkpoint iteration
            for state, policy_sum in EQobserver.sumOfPolicy.items():
                scaled_sum_of_policy = policy_sum / (iteration + 1)  
                policy_sums_over_time[state].append(scaled_sum_of_policy) 

        
    # print("log: action, state, loss",log)
    plot_graph(cum_loss,name="EQasObserver")
    plot_scaled_sum_policy_over_time(policy_sums_over_time, checkpoints,blueAgentActions)               # plot the scaled sum of policy over time for each state
  


    
# EQasAgent()
EQasObserver()
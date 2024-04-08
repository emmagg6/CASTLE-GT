from EnvironmentClass import Environment
from EQApproximationClass import EqApproximation
from ThirdAgentClass import QLearningAgent
from RedAgent import RedAgent

from utility import plot_graph

def EQasAgent():
    """
        blueAgent (EQ approximator) will be the one interacting with the environment
    """
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

    for _ in range(100000000): 
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
    blueAgentActions = ["Analyse","DeployDecoy","Remove","Restore"]                                                      # actions for the blue agent in the cage 4
    EQobserver = EqApproximation(states, num_actions=4, eta=0.1, gamma=0.01)                                             # gamma: exploration; eta: learning rate
    blueAgent = QLearningAgent(num_states, num_actions=4, alpha=0.1, gamma=0.8, epsilon=0.01)                            # gamma: discounting; alpha: learning rate; epsilon: exploration

    cum_loss = []
    log=[]

    for _ in range(100000): 
        state = env.current_state                                                   # get the current state from teh environment
        actionIndex = blueAgent.choose_action(state)                                # let the Q learning agent choose an action
        next_state, loss = env.step(actionIndex)                                    # get the loss for the chosen action
        blueAgent.update_q_table(state, actionIndex,loss,next_state)                # update the policy for the Q learning agent

        # update for EQ Appriximation as observer
        EQobserver.observe_and_update(state, actionIndex, loss)

        cum_loss.append(loss)
        log.append([blueAgentActions[actionIndex],state,loss])


    # print("log: action, state, loss",log)
    plot_graph(cum_loss,name="EQasObserver")


    
EQasAgent()
# EQasObserver()
# from EnvironmentClass import Environment
from EnvironmentBandit import Environment
from EQApproximationClass import EqApproximation
from ThirdAgentClass import QLearningAgent
from RedAgent import RedAgent
from RedAgentBandit import RedAgent

import numpy as np


def EQasAgent(total_iteration=10000, trials = 0):
    print("\nEQ as Agent")
    """
        blueAgent (EQ approximator) will be the one interacting with the environment
    """
    # Initialization 1: Environment
    env = Environment()
    states = env.states

    # Initialization 2: EQ Approximation                                                    # Exploration parameter, γ
    # blueAgentActions = ["Remove","Restore","Sleep"]                                         # for simplicity, the blue agent can only choose between two actions
    # redAgentAction = ["noOp","Attack"]
    
    blueAgentActions = ["Action 2", "Action 3", "Action 4"]                              # Action 2: remove, Action 3: restore, Action 4: sleep
    redAgentActions = ["Action 0", "Action 1"]                                            # Action 0: noOp; Action 1: Attack

    blueAgent =EqApproximation(states, num_actions=3, T=total_iteration)                   # gamma and eta set to 0.01 and 0.1 for simplicity

    # redAgent = RedAgent(5)                                                                  # for simplicity: red agent will attack once for 5 time step 
    redAgent = RedAgent()

    # for plotting and loggin purposes
    training_losses = []
    log=[]
    loss_over_all_actions = {action: 0 for action in blueAgentActions}
    regret = []

    ################# TRAINING #################
    for _ in range(total_iteration): 
        state = env.current_state 
        blueActionIndex = blueAgent.get_action(state)
        # RedActionIndex = redAgent.get_action(redAgentActions)

        blueAction = blueAgentActions[blueActionIndex]
        RedAction = redAgent.get_action(redAgentActions)

        nextstate,loss = env.step(blueAction,RedAction) 
        blueAgent.update_policy(state, blueActionIndex,loss)

        training_losses.append(loss)
        log.append([blueAgentActions[blueActionIndex],state,loss])

        for action in blueAgentActions:
            loss_over_all_actions[action] += env.get_loss(action)
        
        best_current_action = min(loss_over_all_actions, key=loss_over_all_actions.get)
        regret.append(sum(training_losses)-loss_over_all_actions[best_current_action])


    ################# TESTING #################
    testing_losses = []
    for trial in range(trials):
        testing_losses.append([])
        for step in range(20):
            losses = 0

            state = env.current_state 
            blueActionIndex = blueAgent.get_action(state)
            # RedActionIndex = redAsgent.get_action(redAgentActions)

            blueAction = blueAgentActions[blueActionIndex]
            RedAction = redAgent.get_action(redAgentActions)

            nextstate, loss = env.step(blueAction,RedAction)

            losses += loss
        testing_losses[trial].append(losses)

    return training_losses, regret, testing_losses

def EQasObserver(total_iteration=10000, trials = 0):
    print("\nEQ as Observer")
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
    # blueAgentActions = ["Remove","Restore","Sleep"]                                                                     # actions for the blue agent in the cage 4
    blueAgentActions = ["Action 2", "Action 3", "Action 4"]                                                              # actions for the blue agent in Bandit environment
    EQobserver = EqApproximation(states, num_actions=3, T=total_iteration)                                             # gamma: exploration; eta: learning rate
    blueAgent = QLearningAgent(num_states, num_actions=3, alpha=0.1, gamma=0.8, epsilon=0.7)                            # gamma: discounting; alpha: learning rate; epsilon: exploration

    # initialization 3:   
    redAgentAction = ["Action 0", "Action 1"]
    # redAgent = RedAgent(5)      
    redAgent = RedAgent() 

    # for plotting and logging purposes
    losses = []
    log=[]
    policy_sums_over_time = {}  # Prepare dictionary to store the data over time
    checkpoints = []
    checkpoint_frequency = total_iteration * 0.05
    # action_count = {"Remove":0,"Restore":0,"Sleep":0}
    action_count = {"Action 2":0,"Action 3":0,"Action 4":0}

    loss_over_all_actions = {action: 0 for action in blueAgentActions}
    regret = []


    ################# TRAINING #################
    for iteration in range(total_iteration): 
        state = env.current_state                                                       # get the current state from teh environment
        BlueActionIndex = blueAgent.choose_action(state)                                # let the Q learning agent choose an action
        BlueAction = blueAgentActions[BlueActionIndex]

        RedAction = redAgent.get_action(redAgentAction)                                  # let the red agent choose an action (for now, the red agent is not learning)
        next_state, loss = env.step(BlueAction,RedAction)                                    # get the loss for the chosen action
        blueAgent.update_q_table(state, BlueActionIndex,loss,next_state)                     # update the policy for the Q learning agent

        EQobserver.observe_and_update_adaptation(state, BlueAction, loss)

        # ------------------- Logging -------------------
        losses.append(loss)
        log.append([blueAgentActions[BlueActionIndex],state,loss])
        action_count[blueAgentActions[BlueActionIndex]] += 1

        for action in blueAgentActions:
            loss_over_all_actions[action] += env.get_loss(action)
        
        best_current_action = min(loss_over_all_actions, key=loss_over_all_actions.get)
        regret.append(sum(losses)-loss_over_all_actions[best_current_action])


    ################# TESTING #################
    testing_losses_Q = []
    testing_losses_CCE = []

    env_Q = Environment()
    env_CCE = Environment()
    for trial in range(trials):
        testing_losses_Q.append([])
        testing_losses_CCE.append([])
        for step in range(20):
            losses_Q = 0
            losses_CCE = 0

            state_Q = env_Q.current_state 
            state_CCE = env_CCE.current_state
            BlueActionIndex = blueAgent.choose_action(state_Q)
            # RedActionIndex = redAsgent.get_action(redAgentActions)

            BlueAction_Q = blueAgentActions[BlueActionIndex]
            print(f"Q Action: {BlueAction_Q}")
            BlueAction_CCE = EQobserver.get_action_adaptation(state_CCE)
            print(f"CCE Action: {BlueAction_CCE}")
            RedAction = redAgent.get_action(redAgentAction)

            nextstate, loss_Q = env_Q.step(BlueAction_Q, RedAction)
            nextstate, loss_CCE = env_CCE.step(BlueAction_CCE, RedAction)
            print(f"Loss Q: {loss_Q}, Loss CCE: {loss_CCE}")

            losses_Q += loss_Q
            losses_CCE += loss_CCE
        testing_losses_Q[trial].append(losses_Q)
        testing_losses_CCE[trial].append(losses_CCE)
    

    return losses, regret, testing_losses_Q, testing_losses_CCE


TRIALS = 100

lossEQAgent, regretEQAgent, test_lossesEQAgent = EQasAgent(total_iteration=100, trials = TRIALS)
lossEQobserver, regretEQobserver, test_lossesEQobserver_Q, test_lossesEQobserver_CCE = EQasObserver(total_iteration=100, trials = TRIALS)


names_list = ["EXP3-IX", "Agent-Agnostic EXP3-IX"]

# average the losses over the trials, per agent (EQAgent, EQObserver)

average_losses_EQAgent = np.mean(test_lossesEQAgent, axis=0)
average_losses_EQObserver_Q = np.mean(test_lossesEQobserver_Q, axis=0)
average_losses_EQObserver_CCE = np.mean(test_lossesEQobserver_CCE, axis=0)

# standard deviation of the losses over the trials, per agent (EQAgent, EQObserver)

std_losses_EQAgent = np.std(test_lossesEQAgent, axis=0)
std_losses_EQObserver_Q = np.std(test_lossesEQobserver_Q, axis=0)
std_losses_EQObserver_CCE = np.std(test_lossesEQobserver_CCE, axis=0)

# print out mean and standard deviation of the losses over the trials, per agent (EQAgent, EQObserver)

print(f"\n\nAverage losses EXP3-IX: {average_losses_EQAgent},  Standard Deviation: {std_losses_EQAgent}")
print(f"\n\nAverage losses Q-learner: {average_losses_EQObserver_Q},  Standard Deviation: {std_losses_EQObserver_Q}")
print(f"\n\nAverage losses Agent Agnostic EXP3-IX: {average_losses_EQObserver_CCE},  Standard Deviation: {std_losses_EQObserver_CCE}")
print("\n\n\n")


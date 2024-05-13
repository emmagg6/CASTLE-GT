# from EnvironmentClass import Environment
from EnvironmentBandit import Environment
from EQApproximationClass import EqApproximation
from ThirdAgentClass import QLearningAgent
from RedAgent import RedAgent
from RedAgentBandit import RedAgent

from utility import plot_graph
from utility import plot_scaled_sum_policy_over_time
from utility import find_most_favored_action
from utility import plot_regret

import numpy as np


def EQasAgent(total_iteration=10000):
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
    losses = []
    log=[]
    loss_over_all_actions = {action: 0 for action in blueAgentActions}
    regret = []

    for _ in range(total_iteration): 
        state = env.current_state 
        blueActionIndex = blueAgent.get_action(state)
        # RedActionIndex = redAgent.get_action(redAgentActions)

        blueAction = blueAgentActions[blueActionIndex]
        RedAction = redAgent.get_action(redAgentActions)

        nextstate,loss = env.step(blueAction,RedAction) 
        blueAgent.update_policy(state, blueActionIndex,loss)

        losses.append(loss)
        log.append([blueAgentActions[blueActionIndex],state,loss])

        for action in blueAgentActions:
            loss_over_all_actions[action] += env.get_loss(action)
        
        best_current_action = min(loss_over_all_actions, key=loss_over_all_actions.get)
        regret.append(sum(losses)-loss_over_all_actions[best_current_action])

    # Output the policy and visit counts
    print(f"Policy for state '{state}':", blueAgent.eq_approx[state])
    print(f"Visit counts for state '{state}':", blueAgent.visit_count[state])

    # print("log",log)
    # plot_graph(losses,name="EQasAgent")
    return losses,regret

def EQasObserver(total_iteration=10000):
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

        # Print the sum of policy snapshot every fixed iterations
        # if (iteration + 1) % int(checkpoint_frequency) == 0:
        #     # print(f"Iteration {iteration + 1}: Sum of Policy Snapshot:")
        #     checkpoints.append(iteration + 1)  # Record the checkpoint iteration
        #     for state, policy_sum in EQobserver.sumOfPolicy.items(): 
        #         if state not in policy_sums_over_time:
        #             policy_sums_over_time[state] = []  # Initialize it if not present
        #        # Iterate through each state's policy sums
        #         print(f"policy_sum: {policy_sum}",state)
        #         scaled_sum_of_policy = policy_sum[state] / (iteration + 1)  
        #         policy_sums_over_time[state].append(scaled_sum_of_policy) 

        
    # print("log: action, state, loss",log)
    # best_state,best_action,max_value =find_most_favored_action(EQobserver.sumOfPolicy)
    # print(EQobserver.sumOfPolicy)
    # print(f"state {best_state} and action {blueAgentActions[best_action]} has the highset sum of policy value with {max_value}")
    # print("action count" ,action_count)
    # plot_graph(losses,name="EQasObserver")
    # plot_scaled_sum_policy_over_time(policy_sums_over_time, checkpoints, blueAgentActions)   

    # # Output the policy and visit counts
    # print(f"Q-learner: Policy Q-table':\n", blueAgent.q_table)
    # print(f"CCE: Policy for state '{state}':", EQobserver.eq_approx_unknown[str(state)])
    # print(f"CCE: Visit counts for state '{state}':", EQobserver.visit_count_unknown[str(state)])
    return losses,regret


losses_EQAgent = []
losses_EQObserver = []

TRIALS = 100

for trial in range(TRIALS):
    print(f"\n\nTrial {trial+1}")
    lossEQAgent, regretEQAgent = EQasAgent(total_iteration=100)
    lossEQobserver, regretEQobserver = EQasObserver(total_iteration=100)

    # print(f"lossEQAgent: {lossEQAgent}")

    losses_EQAgent.append(sum(lossEQAgent))
    losses_EQObserver.append(sum(lossEQobserver))


names_list = ["EXP3-IX", "Agent-Agnostic EXP3-IX"]

# average the losses over the trials, per agent (EQAgent, EQObserver)

average_losses_EQAgent = np.mean(losses_EQAgent, axis=0)
average_losses_EQObserver = np.mean(losses_EQObserver, axis=0)

# standard deviation of the losses over the trials, per agent (EQAgent, EQObserver)

std_losses_EQAgent = np.std(losses_EQAgent, axis=0)
std_losses_EQObserver = np.std(losses_EQObserver, axis=0)

# print out mean and standard deviation of the losses over the trials, per agent (EQAgent, EQObserver)

print(f"\n\nAverage losses EQAgent: {average_losses_EQAgent},  Standard Deviation: {std_losses_EQAgent}")
print(f"\n\nAverage losses EQObserver: {average_losses_EQObserver},  Standard Deviation: {std_losses_EQObserver}")


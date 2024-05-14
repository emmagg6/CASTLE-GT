# from EnvironmentClass import Environment
from EnvironmentBandit import Environment
from EQApproximationClass import EqApproximation
from ThirdAgentClass import QLearningAgent
from RedAgent import RedAgent
from RedAgentBandit import RedAgent

from utility import plot_graph
from utility import plot_regret


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

    regret_per_state = {}
    regret_per_state_iteration = {}
    loss_per_state_action = {}
    loss_per_state= {}

    for iteration in range(total_iteration): 
        state = env.current_state 
        blueActionIndex = blueAgent.get_action(state)
        # RedActionIndex = redAgent.get_action(redAgentActions)

        blueAction = blueAgentActions[blueActionIndex]
        RedAction = redAgent.get_action(redAgentActions)

        next_state,loss = env.step(blueAction,RedAction) 
        blueAgent.update_policy(state, blueActionIndex,loss)

        losses.append(loss)
        log.append([blueAgentActions[blueActionIndex],state,loss])

        if next_state not in regret_per_state:
            regret_per_state[next_state] = [loss]
            regret_per_state_iteration[next_state]=[iteration]
        else:
            current_total = regret_per_state[next_state][-1]+loss
            regret_per_state[next_state].append(current_total)
            regret_per_state_iteration[next_state].append(iteration)

        # ------------------- Total Regret -------------------
        for action in blueAgentActions:
            loss_over_all_actions[action] += env.get_loss(action)
        
        best_current_action = min(loss_over_all_actions, key=loss_over_all_actions.get)
        regret.append(sum(losses)-loss_over_all_actions[best_current_action])

        # ------------------- Regret per State -------------------
        # getting the actual loss encountered for each state
        if next_state not in loss_per_state:
            loss_per_state[next_state] = [loss]
        else:
            loss_per_state[next_state].append(loss)

        # getting the loss as if the agent had chosen each action
        for action in blueAgentActions:
            action_loss = env.get_loss(action)
            if next_state not in loss_per_state_action:
                loss_per_state_action[next_state] = {action:action_loss} 
            elif action not in loss_per_state_action[next_state]:
                loss_per_state_action[next_state][action]=action_loss
            else:
                loss_per_state_action[next_state][action] += action_loss

        # calculating the regret for this state
        if next_state not in regret_per_state:
            best_loss = min(loss_per_state_action[next_state].values())
            total_loss = sum(loss_per_state[next_state])
            regret_per_state[next_state] = [total_loss - best_loss]
            regret_per_state_iteration[next_state] = [iteration]
        else:
            best_loss = min(loss_per_state_action[next_state].values())
            total_loss = sum(loss_per_state[next_state])
            regret_per_state[next_state].append(total_loss - best_loss)
            regret_per_state_iteration[next_state].append(iteration)



    # Output the policy and visit counts
    print(f"Policy for state '{state}':", blueAgent.eq_approx[state])
    print(f"Visit counts for state '{state}':", blueAgent.visit_count[state])

    # print("log",log)
    # plot_graph(losses,name="EQasAgent")
    return losses,regret, regret_per_state,regret_per_state_iteration

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

    regret_per_state = {}
    regret_per_state_iteration = {}
    loss_per_state_action = {}
    loss_per_state= {}
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

        # ------------------- Total Regret -------------------
        for action in blueAgentActions:
            loss_over_all_actions[action] += env.get_loss(action)

        best_current_action = min(loss_over_all_actions, key=loss_over_all_actions.get)
        regret.append(sum(losses)-loss_over_all_actions[best_current_action])
        
        # ------------------- Regret per State -------------------
        # getting the actual loss encountered for each state
        if next_state not in loss_per_state:
            loss_per_state[next_state] = [loss]
        else:
            loss_per_state[next_state].append(loss)

        # getting the loss as if the agent had chosen each action
        for action in blueAgentActions:
            if next_state not in loss_per_state_action:
                loss_per_state_action[next_state] = {action:env.get_loss(action)} 
            elif action not in loss_per_state_action[next_state]:
                loss_per_state_action[next_state][action]=env.get_loss(action)
            else:
                loss_per_state_action[next_state][action] += env.get_loss(action)

        # calculating the regret for this state
        if next_state not in regret_per_state:
            best_loss = min(loss_per_state_action[next_state].values())
            total_loss = sum(loss_per_state[next_state])
            regret_per_state[next_state] = [total_loss - best_loss]
            regret_per_state_iteration[next_state] = [iteration]
        else:
            best_loss = min(loss_per_state_action[next_state].values())
            total_loss = sum(loss_per_state[next_state])
            regret_per_state[next_state].append(total_loss - best_loss)
            regret_per_state_iteration[next_state].append(iteration)

    print(f"Q-learner: Policy Q-table':\n", blueAgent.q_table)
    print(f"CCE: Policy for state '{state}':", EQobserver.eq_approx_unknown[str(state)])
    print(f"CCE: Visit counts for state '{state}':", EQobserver.visit_count_unknown[str(state)])
    return losses,regret, regret_per_state,regret_per_state_iteration
    
lossEQAgent,regretEQAgent,regretper_stateEQAgent,regretper_stateIterationEQAgent = EQasAgent(total_iteration=100000)
lossEQobserver,regretEQobserver,regretper_stateEQObserver, regretper_stateIterationEQObserver= EQasObserver(total_iteration=100000)

names_list = ["EXP3-IX", "Agent-Agnostic EXP3-IX"]

losses_list = [lossEQAgent, lossEQobserver]
regret_list = [regretEQAgent, regretEQobserver]
regret_per_state_list = [regretper_stateEQAgent,regretper_stateEQObserver]
regret_per_state_iteration_list = [regretper_stateIterationEQAgent,regretper_stateIterationEQObserver]

plot_graph(losses_list, names_list,"Loss")
plot_graph(regret_list, names_list,"Regret")
plot_regret(names_list,regret_per_state_iteration_list,regret_per_state_list)
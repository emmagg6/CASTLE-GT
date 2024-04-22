from EnvironmentClass import Environment
from EQApproximationClass import EqApproximation
from ThirdAgentClass import QLearningAgent
from RedAgent import RedAgent

from utility import plot_graph
from utility import plot_scaled_sum_policy_over_time
from utility import find_most_favored_action


def EQasAgent(total_iteration=10000):
    """
        blueAgent (EQ approximator) will be the one interacting with the environment
    """
    # Initialization 1: Environment
    env = Environment()
    states = env.states

    # Initialization 2: EQ Approximation                                                    # Exploration parameter, γ
    blueAgentActions = ["Remove","Restore","Sleep"]                                         # for simplicity, the blue agent can only choose between two actions
    redAgentAction = ["noOp","Attack"]
    blueAgent =EqApproximation(states, num_actions=3, eta=0.1, gamma=0.7)                   # gamma and eta set to 0.01 and 0.1 for simplicity

    redAgent = RedAgent(5)                                                                  # for simplicity: red agent will attack once for 5 time step 
    
    # for plotting and loggin purposes
    losses = []
    log=[]

    for _ in range(total_iteration): 
        state = env.current_state 
        blueActionIndex = blueAgent.get_action(state)
        RedActionIndex = redAgent.get_action()

        blueAction = blueAgentActions[blueActionIndex]
        RedAction = redAgentAction[RedActionIndex]

        nextstate,loss = env.step(blueAction,RedAction) 
        blueAgent.update_policy(state, blueActionIndex,loss)

        losses.append(loss)
        log.append([blueAgentActions[blueActionIndex],state,loss])

    # Output the policy and visit counts
    print(f"Policy for state '{state}':", blueAgent.eq_approx[state])
    print(f"Visit counts for state '{state}':", blueAgent.visit_count[state])

    # print("log",log)
    # plot_graph(losses,name="EQasAgent")
    return losses
def EQasObserver(total_iteration=10000):
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
    blueAgentActions = ["Remove","Restore","Sleep"]                                                                     # actions for the blue agent in the cage 4
    EQobserver = EqApproximation(states, num_actions=3, eta=0.1, gamma=0.7)                                             # gamma: exploration; eta: learning rate
    blueAgent = QLearningAgent(num_states, num_actions=3, alpha=0.1, gamma=0.8, epsilon=0.7)                            # gamma: discounting; alpha: learning rate; epsilon: exploration

    # initialization 3:   
    redAgentAction = ["NoOp","Attack"]
    redAgent = RedAgent(5)      

    # for plotting and logging purposes
    losses = []
    log=[]
    policy_sums_over_time = {state: [] for state in EQobserver.sumOfPolicy.keys()}  # Prepare dictionary to store the data over time
    checkpoints = []
    checkpoint_frequency = total_iteration * 0.05
    action_count = {"Remove":0,"Restore":0,"Sleep":0}

    for iteration in range(total_iteration): 
        state = env.current_state                                                       # get the current state from teh environment
        BlueActionIndex = blueAgent.choose_action(state)                                # let the Q learning agent choose an action
        BlueAction = blueAgentActions[BlueActionIndex]

        RedActionIndex = redAgent.get_action()
        RedAction = redAgentAction[RedActionIndex]

        next_state, loss = env.step(BlueAction,RedAction)                                    # get the loss for the chosen action
        blueAgent.update_q_table(state, BlueActionIndex,loss,next_state)                     # update the policy for the Q learning agent

        EQobserver.observe_and_update(state, BlueActionIndex, loss)

        # ------------------- Logging -------------------
        losses.append(loss)
        log.append([blueAgentActions[BlueActionIndex],state,loss])
        action_count[blueAgentActions[BlueActionIndex]] += 1

        # Print the sum of policy snapshot every fixed iterations
        if (iteration + 1) % int(checkpoint_frequency) == 0:
            # print(f"Iteration {iteration + 1}: Sum of Policy Snapshot:")
            checkpoints.append(iteration + 1)  # Record the checkpoint iteration
            for state, policy_sum in EQobserver.sumOfPolicy.items():
                scaled_sum_of_policy = policy_sum / (iteration + 1)  
                policy_sums_over_time[state].append(scaled_sum_of_policy) 

        
    # print("log: action, state, loss",log)
    # best_state,best_action,max_value =find_most_favored_action(EQobserver.sumOfPolicy)
    # print(EQobserver.sumOfPolicy)
    # print(f"state {best_state} and action {blueAgentActions[best_action]} has the highset sum of policy value with {max_value}")
    # print("action count" ,action_count)
    # plot_graph(losses,name="EQasObserver")
    plot_scaled_sum_policy_over_time(policy_sums_over_time, checkpoints,blueAgentActions)               # plot the scaled sum of policy over time for each state
    return losses
    
lossEQAgent = EQasAgent(total_iteration=100000)
lossEQobserver = EQasObserver(total_iteration=100000)


losses_list = [lossEQAgent, lossEQobserver]
names_list = ["EQ Agent", "EQ Observer"]
plot_graph(losses_list, names_list)
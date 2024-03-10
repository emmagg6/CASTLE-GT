# Current Env. + Game.

### PLAYERS :

Player 1: Blue Agent

Player 2: Red Agent


### AGENTS :

For this demo, the red agent will be a random agent, and the blue agent will the user (i.e., input).


### ACTION SPACE :

Blue Agent Actions: {'Neutral': 0, 'Block': 1, 'Isolate': 2, 'Unblock': 3, 'Unisolate': 4, 'Restore': 5}

Red Agent Actions: {'Neutral': 0, 'Spread': 1}


### ENVIRONMENTAL DYNAMICS :
> The environment will be a network of hosts and critical servers. The network will be represented as a connection matrix
where each row represents a host or critical server, and each column represents a connection to another host or critical server. 
The connection matrix will be a binary matrix, where 1 represents a connection and 0 represents no connection, and can be thought of as a directed graph.

> The environment will also keep track of the number of infected hosts and critical servers, the load on each host, 
and the number of step connections to critical servers for each host. 

> The environment will also have a set of weights that will be used to calculate the payoffs for each agent based on the changes 
in the environment due to their actions. The weights will be inputs by the user and will be used to calculate the payoffs for each agent.

> The actions available to the agents will be dictated by the updated connection matrix, and sometimes its comparison to the original connection matrix.

> When an action is taken, the connection matrix will be updated to reflect the changes in the environment due to the action.



*actions*

If the blue agent takes the action 'Neutral', the connection matrix will not change.

If the blue agent takes the action 'Block', the agent then selects the host to apply the action to, and the connection matrix will be updated to reflect that the host is now blocked from all critical servers if any are neighbors.

If the blue agent takes the action 'Isolate' the agent then selects the host to apply the action to, the connection matrix will be updated to reflect that the host is now isolated from all of its neighbors in the network.

If the blue agent takes the action 'Unblock', the agent then selects the host to apply the action to, and then any connection between the host and its neighbours that are different than the original connection matrix gets restored as a connection.

If the blue agent takes the action 'Unisolate', the agent then selects the host to apply the action to, and then any connection between the host and its critical neighbours that are different than the original connection matrix gets restored as a connection.

If the blue agent takes the action 'Restore', the agent then selects an infected hosts to applly the action to, and then the host
will be 'uninfected'

The load will be updated to be the cos(1/2 * pi * (number of neighbors / total number of hosts)).

The host step connections to critical servers will be updated to be the (1 - ((minimum number of step connections to critical servers for the host) - 1)/(number of servers)).

If the red agent takes the action 'Neutral', the infected hosts will not change.

If the red agent takes the action 'Spread', the chosen infected hosts will infect its neighbours infected its currently connected neighbours.


### VARIABLES :

H - Total Hosts

h - Hosts nfected

C - Total Critical Servers

c - Critical Servers Infected

psi - Host step connections to critical servers (in the DEMO these are all set to 0 as to not impact the cost)

l = load on host

Omega - Connection Matrix: Host and Critical Server, (binary)

'Neutral' - no change by the player to the network

'Block' - block all access of the select host to critical servers

'Unblock' - unblock all connection of the select host to its original neighbouring host connections

'Isolate' - isolate the selecthost from all of its neighbors in the network

'Unisolate' - reconnect the connection of the select host to its original neighbouring critical connections

'Spread' - spread the infection of select host's neighbor in the network (can be changed to a selection of neighbors)

'Restore' - 'uninfects' the host


### PAYOFFS :

Payoff for Blue Agent = ( # uninfected - good ) - ( Cost on operations )
                      =  w_1 * Delta(H - h) + w_2 * Delta(C - c) - ( w_3 * Delta(psi) + w_4 * Delta(l) )

Payoff for Red Agent = ( utility = infected )
                     = ŵ_1 * Delta(h) + ŵ_2 * Delta(c)


## Current Issues with Pygame Interface / Dynamics

1. Complete the playing loop with the interaction with the pygame interface
2. Fix interface display of available actions and the hosts available for those actions for the user.



    if action_blue is not None and target_blue is not None:
        state.apply_actions((action_blue, action_red), (target_blue, target_red))

        game_interface.draw_network()

        game_interface.draw_text(f"{action_blue} selection on Host {target_blue}", 
                                 (20, HEIGHT//5), align='midleft', color=GREY)
        game_interface.draw_text(f"change in network connectivity of {env.Delta_connections} induced",
                                 (50 HEIGHT//5 + 20), align='midleft', color=LIGHTGREY, font='italic', font_size=15)
        game_interface.draw_text(f"a change in load of {env.l[target_blue]:.2f}  on the connections to the host",
                                 (50, HEIGHT//5 + 33), align='midleft', color=LIGHTGREY, font='italic', font_size=15)
        game_interface.draw_text(f"Adversary chose: {action_red} on Host {target_red}", (20, HEIGHT//5 + 50), align='midleft', color=GREY)
        action_blue, target_blue, action_red, target_red = None, None, None, None  # Reset actions and targets
        state._current_player = 0


        ############## Display Round Score #############
        game_interface.draw_text(f"Round Score (payoff)",
                                 (WIDTH - 100, HEIGHT//5), align='center', font='bold', color=BLACK)
        game_interface.draw_text(f"( {env.reward_blue:.2f}, ",
                                 (WIDTH - 200, HEIGHT//5 + 20), align='center', font='bold', color=BLUE_agent)
        game_interface.draw_text(f"{env.reward_red} )",
                                 (WIDTH - 75, HEIGHT//5 + 20), align='center', font='bold', color=RED_agent)

        ############ Update Running Return ############
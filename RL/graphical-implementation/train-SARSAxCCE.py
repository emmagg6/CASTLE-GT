'''
NOTE: need to activate a different environment (not CybORG)
'''

import numpy as np
import networkx as nx
import math
import random

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.patches import Polygon
from matplotlib import cm
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba

from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial import KDTree

import sklearn
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree

import os




class Voronoi_Diagram :
  '''
  Voronoi Diagram Generation

Generates
> Points on outer circle: $ N \in \mathbb{R}^2 $

> Points about *inner* Gaussian: $ M \in \mathbb{R}^2 $

> Voronoi diagram from both sets

> Visual of diagram
  
  '''
  def __init__(self, N_amnt = 12, N_radius = 1, N_center = (0,0), M_amnt = 6, M_center = 0) :
    self.N_pts = N_amnt
    self.M_pts = M_amnt
    self.N_rad = N_radius
    self.N_mid = N_center
    self.M_mid = M_center


  def generate_points_on_circle(self):
      inner_rad = self.N_rad - self.N_rad/4

      angles = np.arange(self.N_pts) * 2*np.pi/self.N_pts + np.pi/self.N_pts

      # polar coordinates -> Cartesian coordinates
      x_out = self.N_mid[0] + self.N_rad * np.cos(angles)
      y_out = self.N_mid[1] + self.N_rad * np.sin(angles)
      x_in = self.N_mid[0] + inner_rad * np.cos(angles)
      y_in = self.N_mid[1] + inner_rad * np.sin(angles)

      x = np.concatenate((x_out, x_in), axis=0)
      y = np.concatenate((y_out, y_in), axis=0)
      N = np.column_stack((x, y))

      return N

  def generate_points_in_circle(self, M_seed):
      rng = np.random.default_rng(M_seed)

      x, y = rng.normal(loc=self.M_mid, scale=self.N_rad/4, size=(2, self.M_pts))

      M = np.column_stack((x, y))
      
      return M

  def voronoi_mass_set(self, N, M) :
    masses = np.concatenate((N[:,:], M[:,:]))
    return masses

  def voronoi_diag(self, N, M) :
    PTS = self.voronoi_mass_set(N, M)
    vor = Voronoi(PTS)
    return vor, PTS

  def visualize_voronoi(self, VOR, N, M) :
    fig, ax = plt.subplots(figsize = (8, 8))
    voronoi_plot_2d(VOR, ax=ax, show_vertices=False)

    # sites
    ax.plot(N[:, 0], N[:, 1], 'b.', markersize=10)
    ax.plot(M[:, 0], M[:, 1], 'g.', markersize=5)

    # the circles
    N_circle = Circle(self.N_mid, radius=self.N_rad, linestyle='--',
                      edgecolor='b', linewidth = 0.5, facecolor='none')
    N_circle_inner = Circle(self.N_mid, radius=self.N_rad-0.1, linestyle='--',
                      edgecolor='b', linewidth = 0.5, facecolor='none')
    M_circle = Circle(self.N_mid, radius=(self.N_rad/4), linestyle='--',
                      edgecolor='g', linewidth = 0.5, facecolor='none')
    ax.add_patch(N_circle)
    ax.add_patch(N_circle_inner)
    ax.add_patch(M_circle)

    ax.set_xlim([-self.N_rad -1, self.N_rad +1])
    ax.set_ylim([-self.N_rad -1, self.N_rad +1])

    plt.show()



################################### TASK SIMILARITY ###################################

    '''
To interpret $\alpha=0$ and $\alpha=1$ as independent samples...

Here's a possibility: sample two random sets of points $x_0$ and $x_1$ from a 2D normal.  
Interpolate between them as

    x_\alpha = \cos(\frac{\pi}{2} \alpha) \times x_0 + \sin(\frac{\pi}{2} \alpha) \times x_1

for $\alpha\in[0,0.5]$. 

    '''

def combining_tasks(x0, x1, alps) :
  Z = np.zeros((len(alps), len(x0), 2))
  for i, a in enumerate(alps) :
    Z[i] = (np.cos((np.pi / 2) * a) * x0) + (np.sin((np.pi / 2) * a) * x1)
  return Z


############################## SIMILAR LABELING OF VERTICES ##############################


def label_voronoi_sites(original_sites, voronoi, sites):
    # KDTree from the original sites
    original_tree = KDTree(original_sites)

    # closest original site for each site of the current Voronoi diagram
    _, closest_original_sites_indices = original_tree.query(sites)

    # vertices of the voronoi
    vertices = voronoi.vertices

    # distance from each vertex to each site
    distances = cdist(vertices, sites)

    # indices of the three closest sites for each vertex
    closest_sites_indices = np.argpartition(distances, 3, axis=1)[:, :3]

    # initialise one-hot encoded matrix of 0s
    one_hot_matrix = np.zeros((len(vertices), len(original_sites)), dtype=int)

    # for each vertex - set the columns at the indices of the 3 closest sites to 1
    for vertex_index, site_indices in enumerate(closest_sites_indices):
        # site indices -> indices of the closest original sites
        original_site_indices = closest_original_sites_indices[site_indices]

        # Set columns at the original indices of original sites to 1
        one_hot_matrix[vertex_index, original_site_indices] = 1

    return one_hot_matrix



############################ EXTREME POINTS OF VORONOI DIAGRAM ############################

def find_extreme_vertices(vor_diagram):
    vertices = vor_diagram.vertices

    North_index = np.argmax(vertices[:, 1])
    South_index = np.argmin(vertices[:, 1])
    East_index = np.argmax(vertices[:, 0])
    West_index = np.argmin(vertices[:, 0])

    North_coord = vertices[North_index]
    South_coord = vertices[South_index]
    East_coord = vertices[East_index]
    West_coord = vertices[West_index]

    return North_index, North_coord, South_index, South_coord, East_index, East_coord, West_index, West_coord



############################ VORONOI TO GRAPH ############################

def voronoi_to_graph(vor, labels):
    G = nx.Graph()  # NetworkX graph

    for idx, vertex in enumerate(vor.vertices):
        label_str = str(labels[idx].tolist())  # one-hot list to string
        G.add_node(label_str, pos=vertex.tolist())

    for edge in vor.ridge_vertices:
        if edge[0] != -1 and edge[1] != -1:
            label_str_0 = str(labels[edge[0]].tolist())
            label_str_1 = str(labels[edge[1]].tolist())
            G.add_edge(label_str_0, label_str_1)

    return G

def draw_nx_graph(G):
    pos = nx.get_node_attributes(G, 'pos')
    labels = {node: node for node in G.nodes()}

    plt.figure(figsize=(8, 8))
    nx.draw_networkx(G, pos, with_labels=False, node_color='white', node_size = 50, edge_color='grey')
    nx.draw_networkx_labels(G, pos, labels, font_size=5)

    plt.show()


    ############################### STATE-ACTION PAIRS ###############################


def neighbors(graph):
    """
    mapping from each node to its neighbors
    sorted in clockwise order relative to the node's upward direction
    """
    adjacency_dict = {}

    for node in graph.nodes():
        x1, y1 = graph.nodes[node]['pos']

        neighbors = list(graph.neighbors(node))
        neighbors = [n for n in graph.neighbors(node) if n != node]

        # sorting the neighbors in clockwise order relative to the top (y-values)
        neighbors.sort(key=lambda n:
            - (math.atan2(graph.nodes[n]['pos'][1] - y1, graph.nodes[n]['pos'][0] - x1) - math.pi / 2) % (2 * math.pi))
        
                # calculating the Euclidean distance for each neighbor and storing it along with the neighbor
        neighbors_with_distance = [(n, np.linalg.norm(np.array(graph.nodes[n]['pos']) - np.array(graph.nodes[node]['pos']))) for n in neighbors]

        adjacency_dict[node] = neighbors_with_distance

    return adjacency_dict


########################## VISUALS ##########################


def get_index(graph, coordinates):
    for node, data in graph.nodes(data=True):
        if np.array_equal(data['pos'], coordinates):
            return node
    raise ValueError("Coordinates not found in graph.")



def visualize_path(G, path_indices = None, start_coordinates = None, goal_coordinates = None, see_labels = False):
    """ 
    Visualize a path in a networkx graph - instead of voronoi

    """
    pos = nx.get_node_attributes(G, 'pos')

    start_index = get_index(G, start_coordinates) if start_coordinates is not None else None
    goal_index = get_index(G, goal_coordinates) if goal_coordinates is not None else None

    plt.figure(figsize=(8, 8))

    nx.draw_networkx_nodes(G, pos, node_color='white', node_size=0)
    nx.draw_networkx_edges(G, pos, edge_color='grey')

    if path_indices is not None:
        path_edges = list(zip(path_indices[:-1], path_indices[1:]))
        
        includes_nodes = all(G.has_node(node) for node in path_indices)
        includes_edges = all(G.has_edge(*edge) for edge in path_edges)

        colors = [cm.Reds(0.35 + 0.5*(len(path_edges)-i)/len(path_edges)) for i in range(len(path_edges))]

        if includes_nodes and includes_edges : 
            for i in range(len(path_edges)):
                nx.draw_networkx_edges(G, pos, edgelist=[path_edges[i]], edge_color=colors[i], width=2)

            for i, node in enumerate(path_indices):
                if node == start_index:  # Skip the start node
                    continue
                node_color = np.array(colors[min(i, len(colors)-1)]).reshape(1, -1)
                nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color=node_color, node_size=50)

                if see_labels == True :
                    rgba = to_rgba(node_color)
                    label_color = 'white' if rgba[0] < 0.25 else 'black'
                    nx.draw_networkx_labels(G, pos, {node: node}, font_color=label_color, font_size=5)
                    nx.draw_networkx_labels(G, pos, {node: i}, font_color=label_color, font_size=5)

            if start_index is not None:
                nx.draw_networkx_nodes(G, pos, nodelist=[start_index], node_color='blue', node_size=150, alpha=0.3)
                nx.draw_networkx_labels(G, pos, {start_index: start_index}, font_color='black', font_size=0)

            if goal_index is not None:
                nx.draw_networkx_nodes(G, pos, nodelist=[goal_index], node_shape='*', node_color='yellow', 
                                       node_size=300, edgecolors='black')
                nx.draw_networkx_labels(G, pos, {goal_index: goal_index}, font_color='black', font_size=0)

    plt.title('Path on Original Graph')

    legend_elements = [
        Line2D([0], [0], marker='o', color='blue', lw=0, markersize=10, alpha=0.3, label='Start'),
        Line2D([0], [0], marker='*', color='yellow', markeredgecolor='black', markersize=10, label='Goal')
    ]
    plt.legend(handles=legend_elements, loc='lower left', fontsize='x-small')

    plt.show()




############################## AGENT INITIALIZATIONS ##############################


actions_main = ['left', 'right', 'back']
actions_edge = ['forward', 'back']
actions_four = ['left', 'right', 'forward', 'back']
actions_five = ['left', 'right',  'back', 'forward_left', 'forward_right']
actions_six = ['left', 'right', 'forward_left', 'forward_right', 'forward', 'back']

def create_q_table(state_actions_dist, init_value=1.0):
    q_table = {}
    for state, actions_dist in state_actions_dist.items():
        actions = [ad[0] for ad in actions_dist]  # Extract actions from tuples
        if len(actions) == 2 :
            acts = actions_edge
        elif len(actions) == 3 :
            acts = actions_main
        # after 3 in the original did not actually occur in any trials so far 
        # only occurs in the slight shifting - but just in case:
        elif len(actions) == 4 :
            acts = actions_four
        elif len(actions) == 5 :
            acts = actions_five
        elif len(actions) == 6 :
            acts = actions_six

        for i, action in enumerate(actions):
            q_table[(state, acts[i])] = init_value

    return q_table



############################### TRAINING #################################

'''
SARSA Q-table and Agent-Agnostic EXP3-IX algorithm training

Note: actions are a string given by a rotation system -- see paper for explanation

    *OPTIONAL CAPABILITY*
RETRAINING / DIFFERENT GRAPH : if the Q-table has been pretrained and already 
contains some knowledge about the environment, setting a high initial Q-value 
for new state-action pairs could potentially cause some issues. In particular, 
if the new state-action pairs are actually not very good (i.e., they lead to 
states with low rewards), then initializing them with high Q-values could cause 
the agent to prefer these new, unexplored actions over the actions it has already 
learned to be good. This could lead to suboptimal behavior until the Q-values of 
the new state-action pairs are updated downwards.
'''

def get_index(graph, coordinates):
    for node, data in graph.nodes(data=True):
        if np.array_equal(data['pos'], coordinates):
            return node
    raise ValueError("Coordinates not found in graph.")


def action_selection(graph, state_actions_dist, q_table, state, previous_state, epsilon=0.1):
    """
    Select an action from the state using epsilon-greedy action selection.
    """
    # Extract the actions from (action, distance) pairs
    state_actions = {state: [action_dist[0] for action_dist in action_dists] for state, action_dists in state_actions_dist.items()}

    if np.random.rand() < epsilon:
        # Action going back
        back = [a for a in range(len(state_actions[state])) if state_actions[state][a] == previous_state][0]
        # print('back', back)

        # Action left & right
        if len(state_actions[state]) == 2:
            forward = (back + 1) % len(state_actions[state])
            viable_actions = {'forward': forward, 'back': back}
        elif len(state_actions[state]) == 3 :
            left = (back + 1) % len(state_actions[state])
            right = (back - 1) % len(state_actions[state])
            # Viable actions is a dictionary from 'left', 'right', and 'backward' and are mapped to their back, left, and right indices in the adjacency dict
            viable_actions = {'left': left, 'right': right, 'back': back}
        elif len(state_actions[state]) == 4 :
            left = (back + 1) % len(state_actions[state])
            right = (back - 1) % len(state_actions[state])
            forward = (back + 2) % len(state_actions[state])
            # Viable actions is a dictionary from 'left', 'right', and 'backward' and are mapped to their back, left, and right indices in the adjacency dict
            viable_actions = {'left': left, 'right': right, 'forward': forward, 'back': back}
        elif len(state_actions[state]) == 5 :
            left = (back + 1) % len(state_actions[state])
            right = (back - 1) % len(state_actions[state])
            forward_left = (back + 2) % len(state_actions[state])
            forward_right = (back + 3) % len(state_actions[state])
            # Viable actions is a dictionary from 'left', 'right', and 'backward' and are mapped to their back, left, and right indices in the adjacency dict
            viable_actions = {'left': left, 'right': right, 'forward_left': forward_left, 'forward_right': forward_right, 'back': back}
        else :
            left = (back + 1) % len(state_actions[state])
            right = (back - 1) % len(state_actions[state])
            forward_left = (back + 2) % len(state_actions[state])
            forward_right = (back - 2) % len(state_actions[state])
            forward = (back + 3) % len(state_actions[state])
            # Viable actions is a dictionary from 'left', 'right', and 'backward' and are mapped to their back, left, and right indices in the adjacency dict
            viable_actions = {'left': left, 'right': right, 'forward_left': forward_left, 'forward_right': forward_right, 'forward': forward, 'back': back}



        # geting the list of actions and their corresponding Q-values
        actions = list(viable_actions.keys())
        q_values = [q_table[(state, a)] for a in actions]

        # normalize the Q-values to form a probability distribution
        soft_q_values = np.exp(q_values)  # use softmax to ensure the Q-values are positive
        prob_distribution = soft_q_values / np.sum(soft_q_values)

        # select an action based on the Q-values
        action = np.random.choice(actions, p=prob_distribution)
        index = viable_actions[action]


    # greedy action selection :
    else:
        # Action going back
        back = [a for a in range(len(state_actions[state])) if state_actions[state][a] == previous_state][0]

        # Action left & right
        if len(state_actions[state]) == 2:
            forward = (back + 1) % len(state_actions[state])
            viable_actions = {'forward': forward, 'back': back}
        elif len(state_actions[state]) == 3 :
            left = (back + 1) % len(state_actions[state])
            right = (back - 1) % len(state_actions[state])
            # Viable actions is a dictionary from 'left', 'right', and 'backward' and are mapped to their back, left, and right indices in the adjacency dict
            viable_actions = {'left': left, 'right': right, 'back': back}
        elif len(state_actions[state]) == 4 :
            left = (back + 1) % len(state_actions[state])
            right = (back - 1) % len(state_actions[state])
            forward = (back + 2) % len(state_actions[state])
            # Viable actions is a dictionary from 'left', 'right', and 'backward' and are mapped to their back, left, and right indices in the adjacency dict
            viable_actions = {'left': left, 'right': right, 'forward': forward, 'back': back}
        elif len(state_actions[state]) == 5 :
            left = (back + 1) % len(state_actions[state])
            right = (back - 1) % len(state_actions[state])
            forward_left = (back + 2) % len(state_actions[state])
            forward_right = (back + 3) % len(state_actions[state])
            # Viable actions is a dictionary from 'left', 'right', and 'backward' and are mapped to their back, left, and right indices in the adjacency dict
            viable_actions = {'left': left, 'right': right, 'forward_left': forward_left, 'forward_right': forward_right, 'back': back}
        else :
            left = (back + 1) % len(state_actions[state])
            right = (back - 1) % len(state_actions[state])
            forward_left = (back + 2) % len(state_actions[state])
            forward_right = (back - 2) % len(state_actions[state])
            forward = (back + 3) % len(state_actions[state])
            # Viable actions is a dictionary from 'left', 'right', and 'backward' and are mapped to their back, left, and right indices in the adjacency dict
            viable_actions = {'left': left, 'right': right, 'forward_left': forward_left, 'forward_right': forward_right, 'forward': forward, 'back': back}


        # Select the action with the highest Q-value
        action = max(viable_actions, key=lambda a: q_table[(state, a)])
        index = viable_actions[action]

    return action, index


def train(graph, goal_coord, method, q_table, state_actions_dist, episodes=5000, alpha=0.1, gamma=0.9, initial_epsilon=0.1, min_epsilon=0.01, max_distance=500, initial_q=1):
    """
    Train the agent using SARSA with e-greedy action selection.
    Update Agent-Agnostic EXP3-IX algorithm during training
    """
    Q = q_table.copy()
    Qs = []

    cce = {}

    # Initialize new states and actions in the Q-table
    for state, action_dists in state_actions_dist.items():
        actions = [ad[0] for ad in action_dists]  # Extract actions (neighbor labels) from tuples

        # Determine the action strings based on the number of actions
        if len(actions) == 2:
            acts = actions_edge
        elif len(actions) == 3:
            acts = actions_main
        elif len(actions) == 4:
            acts = actions_four
        elif len(actions) == 5:
            acts = actions_five
        elif len(actions) == 6:
            acts = actions_six

        # Check if all action strings for this state are in the Q-table
        for i, action in enumerate(actions):
            if (state, acts[i]) not in Q:
                Q[(state, acts[i])] = initial_q

    epsilon = initial_epsilon
    goal = get_index(graph, goal_coord)

    for episode in range(episodes):
        state = np.random.choice(list(graph.nodes))
        previous_state, _ = random.choice(state_actions_dist[state])
        distance = 0

        # Initial Action selection
        action, action_idx = action_selection(graph, state_actions_dist, Q, state, previous_state, epsilon=epsilon)

        while state != goal and distance < max_distance:

            # State update based on action
            new_state, dist = state_actions_dist[state][action_idx]

            # Action selection for next action (for SARSA update rule)
            next_action, next_action_idx = action_selection(graph, state_actions_dist, Q, new_state, state, epsilon=epsilon)

            distance += dist

            reward = 0 if new_state != goal else (1 - distance)

            # SARSA Update Rule
            Q[(state, action)] = Q[(state, action)] + alpha * (reward + gamma * Q.get((new_state, next_action), initial_q) - Q[(state, action)])
            # CCE Update 
            cce = update_eq(cce, state, action, reward, T = episodes)

            action = next_action
            action_idx = next_action_idx
            previous_state = state
            state = new_state

        # Decaying epsilon with episodes
        epsilon = max(min_epsilon, epsilon / np.sqrt((episode + 1)))

        if (episode + 1) % (episodes / 5) == 0:
            Qs.append(Q)

    return Qs, cce

def initialize_or_update_cce(cce, state, action):
    # s = tuple(state)
    s = state
    if s not in cce:
        cce[s] = {}

    if action not in cce[s]:
        # Check if there are any actions already for this state
        if cce[s]:  # If there are existing actions
            # Initialize as the average approx for all actions for this state, and visit count as 0
            avg_approx = np.mean([cce[s][a][0] for a in cce[s]])
        else:
            # If no actions exist, initialize to 1.0
            avg_approx = 1.0
        cce[s][action] = [avg_approx, 0]

    return cce

def update_eq(cce, state, action, loss, T=100):
    # state = tuple(state)
    cce = initialize_or_update_cce(cce, state, action)

    eq_approx_dict = {act: val[0] for act, val in cce[state].items()}
    visit_count_dict = {act: val[1] for act, val in cce[state].items()}
    total_eq_approx = sum(eq_approx_dict.values())
    policy = {act: eq_approx_dict[act] / total_eq_approx for act in eq_approx_dict}

    # Hyperparameter updates:
    gamma = np.sqrt(2 * np.log(len(policy)) / (len(policy) * T))
    eta = 2 * gamma

    estimated_loss = loss / (policy[action] + gamma)

    log_probs = {act: np.log(policy[act] + 1e-50) for act in policy}
    log_probs[action] -= eta * estimated_loss

    max_log_prob = max(log_probs.values())
    for act in eq_approx_dict:
        eq_approx_dict[act] = np.exp(log_probs[act] - max_log_prob)
        cce[state][act][0] = eq_approx_dict[act]

    visit_count_dict[action] += 1
    cce[state][action][1] = visit_count_dict[action]

    return cce



############################### TESTING #################################

def softmax(q_values, temperature=0.5) :
    q_values = np.array(q_values) 
    e_q = np.exp(q_values / temperature)
    return e_q / e_q.sum(axis=0)


def test(cce, q_table, state_actions_dist, graph, start_coordinates, goal_coordinates, zeta = 1000, max_steps=10000, temperature=0.5):
    start_index = get_index(graph, start_coordinates)
    goal_index = get_index(graph, goal_coordinates)

    state = start_index
    path_indices = [start_index]
    path_coords = [graph.nodes[start_index]['pos']]

    previous_state = None
    total_distance = 0

    num_sarsa_actions = 0
    num_cce_actions = 0

    for step in range(max_steps):
        # Get the viable actions according to the adjacency dictionary for the current state
        if len(state_actions_dist[state]) == 2:
            valid_actions = actions_edge
        elif len(state_actions_dist[state]) == 3:
            valid_actions = actions_main
        elif len(state_actions_dist[state]) == 4:
            valid_actions = actions_four
        elif len(state_actions_dist[state]) == 5:
            valid_actions = actions_five
        else:
            valid_actions = actions_six

        # Check for valid actions
        if not valid_actions:
            print(f"No valid actions at state {state}")
            print(f"Number of state-actions: {len(state_actions_dist[state])}")
            break

        # CCE OR Q-VALUE SELECTION
        # Check if the state is in the CCE table

        if state in cce:
            visit_count_dict = {act: val[1] for act, val in cce[state].items()}
            if np.sum(list(visit_count_dict.values())) > zeta:
                eq_approx_dict = {act: val[0] for act, val in cce[state].items()}
                # action_probs = softmax(list(eq_approx_dict.values()), temperature)
                # action = np.random.choice(list(eq_approx_dict.keys()), p=action_probs)
                action = max(eq_approx_dict, key=eq_approx_dict.get)
                action_idx = valid_actions.index(action)
                num_cce_actions += 1
        else :
            # Q-values for each valid action
            q_values = [q_table.get((state, action), -1) for action in valid_actions]
            # Action selection using softmax probabilities
            action_probs = softmax(q_values, temperature)
            action = np.random.choice(valid_actions, p=action_probs)
            # Find the index of the chosen action
            action_idx = valid_actions.index(action)
            num_sarsa_actions += 1

        new_state, dist = state_actions_dist[state][action_idx]

        total_distance += dist

        path_indices.append(new_state)
        path_coords.append(graph.nodes[new_state]['pos'])

        # Check if the goal is reached
        if new_state == goal_index:
            cce_precentage = num_cce_actions / (num_cce_actions + num_sarsa_actions)
            break

        previous_state = state
        state = new_state

    return path_coords, path_indices, total_distance, cce_precentage


############ INITIALIZING ############

TRIALS = 100

num_inner_sites = 7
temp = 0.01

ALL_Q_TABLES = []
ALL_CCE = []
ALL_GRAPHS = []

############################## START LOOP ##############################

### Create Environment ###
VOR = Voronoi_Diagram(M_amnt = num_inner_sites)
Y = VOR.generate_points_on_circle()

### Trials ###
for trial in range(TRIALS) :
    # if trial % 10 == 0 :
    print(f'Trial {trial} of {TRIALS}')

    star = 0
    hex = 0

    x_0 = VOR.generate_points_in_circle(M_seed = 0)

    dia, sites = VOR.voronoi_diag(Y, x_0)
    voronoi_diagram = dia 
    voronoi_diagrams_sites = sites


    label = label_voronoi_sites(voronoi_diagrams_sites, voronoi_diagram, voronoi_diagrams_sites)
    graph = voronoi_to_graph(voronoi_diagram, label)

    North_index, North_coord, South_index, South_coord, East_index, East_coord, West_index, West_coord = find_extreme_vertices(voronoi_diagram)

    state_actions_dist = neighbors(graph)

    # TRAINING ON VORONOI DIAGRAM

    init_q_table = create_q_table(state_actions_dist, init_value=1.0)

    q_table, cce = train(graph = graph,
                                goal_coord = East_coord,
                                method = 'egreedy',
                                q_table = init_q_table,
                                state_actions_dist = state_actions_dist,
                                initial_epsilon = 0.25,
                                episodes = 5000)
    
    ALL_Q_TABLES.append(q_table)
    ALL_CCE.append(cce)
    ALL_GRAPHS.append(graph)

###################### SAVE OUTPUTS ######################

import pickle

folder = '/scratch/egraham/CASTLE-GT/RL/graphical-implementation/runs'
os.makedirs(folder, exist_ok=True)

with open(os.path.join(folder, 'SARSA_q_tables.pkl'), 'wb') as f:
    pickle.dump(ALL_Q_TABLES, f)

with open(os.path.join(folder, 'SARSA_cce.pkl'), 'wb') as f:
    pickle.dump(ALL_CCE, f)

with open(os.path.join(folder, 'graphs.pkl'), 'wb') as f:
    pickle.dump(ALL_GRAPHS, f)

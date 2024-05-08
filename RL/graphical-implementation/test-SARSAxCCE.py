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

# from scipy.spatial import Voronoi, voronoi_plot_2d
# from scipy.spatial import KDTree

# import sklearn
# from sklearn.preprocessing import MinMaxScaler
# from scipy.spatial.distance import cdist
# from scipy.spatial import KDTree

import os

from runs.loading import loading

folder = '/scratch/egraham/CASTLE-GT/RL/graphical-implementation/runs'
load = loading()
load.load_train(path = folder)


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

def find_extreme_vertices(graph):
    nodes = np.array([graph.nodes[node]['pos'] for node in graph.nodes()])

    ordered_nodes_xaxis = nodes[nodes[:, 0].argsort()]

    East_index = np.argmax(ordered_nodes_xaxis[:, 0])
    East_coord = ordered_nodes_xaxis[East_index]

    West_index = np.argmin(ordered_nodes_xaxis[:, 0])
    West_coord = ordered_nodes_xaxis[West_index]

    return East_index, East_coord, West_index, West_coord



############################ VORONOI EXTREMES AND TO GRAPH ############################

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
            else:
                # Q-values for each valid action
                q_values = [q_table.get((state, action), -1) for action in valid_actions]
                action_probs = softmax(q_values, temperature)
                action = np.random.choice(valid_actions, p=action_probs)
                action_idx = valid_actions.index(action)
                num_sarsa_actions += 1
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
            break

        previous_state = state
        state = new_state

    cce_precentage = num_cce_actions / (num_cce_actions + num_sarsa_actions)
    return path_coords, path_indices, total_distance, cce_precentage


############ INITIALIZING ############

TRIALS = 100

zetas_small = np.arange(200000, 220000, 10000)
zetas_mid = np.arange(220000, 250000, 1000)
zetas_large = np.arange(250000, 310000, 10000)
zetas = np.concatenate((zetas_small, zetas_mid, zetas_large))
print(zetas)

num_inner_sites = 7
temp = 0.01

ALL_DISTS = []
ALL_PATHS = []
ALL_CCE_PRECENTAGES = []
ALL_ZETAS = []

Q_tables = load.Q_tables
CCEs = load.cces
graphs = load.graphs

############################## START LOOP ##############################

### Trials ###
for trial in range(TRIALS) :
    print(f'Trial {trial}')
    ALL_PATHS.append([])
    ALL_DISTS.append([])
    ALL_CCE_PRECENTAGES.append([])
    ALL_ZETAS.append([])

    cce =CCEs[trial]
    q_table = Q_tables[trial]
    graph = graphs[trial]

    East_index, East_coord, West_index, West_coord = find_extreme_vertices(graph)
    state_actions_dist = neighbors(graph)


    # TESTING PATH AND DISTANCE MEASURE AFTER TRAINING
    for zeta in zetas:
        print(f'Testing with zeta = {zeta}')
        path_indices, path_coords, path_dist, propCCE = test(cce, q_table, state_actions_dist, graph, 
                                                        West_coord, East_coord, 
                                                        zeta, max_steps = 100000, temperature = 0.1)

        ALL_PATHS[trial].append(path_coords)
        ALL_DISTS[trial].append(path_dist)
        ALL_ZETAS[trial].append(zeta)
        ALL_CCE_PRECENTAGES[trial].append(propCCE)

###################### SAVE OUTPUTS ######################

import pickle

os.makedirs(folder, exist_ok=True)


with open(os.path.join(folder, 'part_TEST_dists.pkl'), 'wb') as f:
    pickle.dump(ALL_DISTS, f)

with open(os.path.join(folder, 'part_TEST_paths.pkl'), 'wb') as f:
    pickle.dump(ALL_PATHS, f)


with open(os.path.join(folder, 'part_TEST_cce_percentages.pkl'), 'wb') as f:
    pickle.dump(ALL_CCE_PRECENTAGES, f)

with open(os.path.join(folder, 'part_TEST_zetavalues.pkl'), 'wb') as f:
    pickle.dump(zetas, f)

with open(os.path.join(folder, 'part_TEST_all_zetas.pkl'), 'wb') as f:
    pickle.dump(ALL_ZETAS, f)
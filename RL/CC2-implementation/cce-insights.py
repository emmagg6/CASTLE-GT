'''
Looking into a CCE approximation

For reference, this is the format of the cce dictionary:
cce = {
    state: [eq_approx, visit_count]
}

where:
    eq_approx: np.array() of the equilibrium approximations for each action in the state
    visit_count: np.array() of the number of visits to each action in the state

For example:
cce[state][0][action] = 0.5, and its corresponding visit count is, say, cce[state][1][action] = 100
'''

import numpy as np
import os
from Agents.BlueAgents.ApproxCCE import CCE

model_dir = "gt"
model_file_GT = "10000cce.pkl"

# Load the CCE approximation
cce = CCE()
ckpt = os.path.join(os.getcwd(), "Models", model_dir, model_file_GT)
cce.load_eq(ckpt)

# Final Action Space
action_space = [133, 134, 135, 139]     # restore enterprise and opserver
action_space += [3, 4, 5, 9]            # analyse enterprise and opserver
action_space += [16, 17, 18, 22]        # remove enterprise and opserer
action_space += [11, 12, 13, 14]        # analyse user hosts
action_space += [141, 142, 143, 144]    # restore user hosts
action_space += [132]                   # restore defender
action_space += [2]                     # analyse defender
action_space += [15, 24, 25, 26, 27]    # remove defender and user hosts
action_space = action_space + [51] + [116] + [55] + [37, 61, 43, 130, 91, 44, 38, 115, 54, 107, 76, 131, 106, 28, 35, 120, 90, 119, 102, 29, 113, 126]

action_index = {action: i for i, action in enumerate(action_space)}


# Get the CCE approximation for a state
state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
s = tuple(state)
eq, visits = 0, 1
print("State: ", state)
# print(f"CCE approximations: {cce.cce[s][eq]}")
# print("CCE visits: ", cce.cce[s][visits])

# Wholistically, how many states have been visited?
print()
print("Overview: \nHow many states have been visited?")
states = list(cce.cce.keys())
print("Number of states visited: ", len(states))

# How many states have non-zero equilibria?
print("How many states have non-zero equilibria?")
non_zero_eq = 0
for s in states:
    if np.sum(cce.cce[s][eq]) > 0:
        non_zero_eq += 1
print("Number of states with non-zero equilibria: ", non_zero_eq)

# what is the hightest equilibrium value?
print()
print("What is the highest/lowest/avg equilibrium value?")
highest_eq = 0
for s in states:
    eq = cce.cce[s][0]
    if np.max(eq) > highest_eq:
        highest_eq = np.max(eq)
print("Highest equilibrium value: ", highest_eq)

# what is the lowest equilibrium value? what is the average equilibrium value?
lowest_eq = 100000
for s in states:
    eq = cce.cce[s][0]
    if np.min(eq) < lowest_eq:
        lowest_eq = np.min(eq)
average_eq = 0
for s in states:
    eq = cce.cce[s][0]
    average_eq += np.sum(eq)
average_eq /= len(states)*len(action_space)
print("Lowest equilibrium value: ", lowest_eq)
print("Average equilibrium value: ", average_eq)

# what is the highest, lowest, and average visit counts for each state?
print()
print("What is the highest/lowest/avg visit count for each state?")
highest_visits = 0
lowest_visits = 100000
avereage_visits = 0
for s in states:
    visits = cce.cce[s][1]
    if np.max(visits) > highest_visits:
        highest_visits = np.max(visits)
    if np.min(visits) < lowest_visits:
        lowest_visits = np.min(visits)
    avereage_visits += np.sum(visits)
avereage_visits /= len(states)*len(action_space)
print("Highest visit count: ", highest_visits)
print("Lowest visit count: ", lowest_visits)


# for a specific action, what is the average equilibiium value and visit count?
print()
print("For a specific action, what is the average equilibrium value and visit count?")
action = 51
action_idx = action_index[action]
action_eq = []
action_visits = []
eq, visits = 0, 1
print("Action: ", action, "Action Index: ", action_idx)
for s in states:
    if action in action_index:
        action_eq.append(cce.cce[s][eq][action_idx])
        action_visits.append(cce.cce[s][visits][action_idx])
print(f"Average equilibrium value for action {action}: ", np.mean(action_eq))
print(f"Average visit count for action {action}: ", np.mean(action_visits))

# for a specific action, what proportion of states was this action visited?
print()
print("For a specific action, what proportion of states was this action visited?")
action = 126
action_idx = action_index[action]
eq, visits = 0, 1
print("Action: ", action, "Action Index: ", action_idx)
action_visits = []
for s in states:
    if action in action_index:
        action_visits.append(cce.cce[s][visits][action_idx])
print(f"Proportion of states visited for action {action}: {len(action_visits) / len(states):e}")


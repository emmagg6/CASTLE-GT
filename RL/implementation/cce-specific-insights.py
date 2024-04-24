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
from Agents.BlueAgents.ApproxCCEspecific import CCE

model_dir = "gt-specific"
model_file_GT = "10000cce.pkl"

# Load the CCE approximation
approx = CCE()
ckpt = os.path.join(os.getcwd(), "Models", model_dir, model_file_GT)
approx.load_eq(ckpt)


# Get the CCE approximation for a state
state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
s = tuple(state)
print("State: ", state)
states_eq = [approx.cce[s][a][0] for a in approx.cce[s]]
states_visits = [approx.cce[s][a][1] for a in approx.cce[s]]
print
print(f"CCE approximations: {states_eq}")
print("CCE visits: ", states_visits)

# Wholistically, how many states have been visited?
states = list(approx.cce.keys())
print("Number of states visited: ", len(states))

# How many states have non-zero equilibria?
non_zero_eq = 0
for s in states:
    for eq in [approx.cce[s][a][0] for a in approx.cce[s]]:
        if np.sum(eq) > 0:
            non_zero_eq += 1
print("Number of states with non-zero equilibria: ", non_zero_eq)

# Average equilibrium value
average_eq = 0
num = 0
for s in states:
    for eq in [approx.cce[s][a][0] for a in approx.cce[s]]:
        average_eq += np.sum(eq)
        num += 1
average_eq = average_eq / num
print("Average equilibrium value: ", average_eq)

# Average equilibrium value for states with over 100 visits
average_eq = 0
num = 0
balance = 1000
for s in states:
    for a in approx.cce[s]:
        if approx.cce[s][a][1] > balance:
            for eq in [approx.cce[s][a][0] for a in approx.cce[s]]:
                average_eq += np.sum(eq)
                num += 1
average_eq = average_eq / num
print(f"Average equilibrium value for states with over {balance} visits: ", average_eq)

# what is the hightest equilibrium value?
highest_eq = 0
for s in states:
    for eq in [approx.cce[s][a][0] for a in approx.cce[s]]:
        if np.max(eq) > highest_eq:
            highest_eq = np.max(eq)
print("Highest equilibrium value: ", highest_eq)

# what is the highest, lowest, and average visit counts for each state?
highest_visits = 0
lowest_visits = 100000
avereage_visits = 0
num = 0
for s in states:
    for cnt in [approx.cce[s][a][1] for a in approx.cce[s]]:
        if np.max(cnt) > highest_visits:
            highest_visits = np.max(cnt)
        if np.min(cnt) < lowest_visits:
            lowest_visits = np.min(cnt)
        avereage_visits += cnt
        num +=1
avereage_visits = avereage_visits / num
print("Highest visit count: ", highest_visits)
print("Lowest visit count: ", lowest_visits)

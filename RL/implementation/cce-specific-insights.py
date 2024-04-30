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
from Agents.BlueAgents.ApproxCCEv2 import CCE

######## Select Model to Check ########
model_dir = "cce-bline"
# model_dir = "gt-specific"                 # b-line == greedy agent
# model_dir = "gt-specific-meander"         # meander == random agent
# model_dir = "gt-specific-sleep"             # sleep == no action agent
#######################################

model_file_GT = "10000cce.pkl"

# Load the CCE approximation
approx = CCE()
ckpt = os.path.join(os.getcwd(), "Models", model_dir, model_file_GT)
approx.load_eq(ckpt)

print("Evaulation of the CCE approximation for" + model_dir)


# Get the CCE approximation for a state
state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
s = tuple(state)
states_eq = [approx.cce[s][a][0] for a in approx.cce[s]]
states_visits = [approx.cce[s][a][1] for a in approx.cce[s]]
print("What are the CCE approximations for a state?\n")
print("State: ", state)
print(f"\nCCE approximations: {states_eq}")
print(f"\nWhat about the visits for each action in the state?")
print("CCE visits: ", states_visits)
print("\n\n")

# Wholistically, how many states have been visited?
print("How many states have been visited?")
states = list(approx.cce.keys())
print("Number of states visited: ", len(states))

# How many states have non-zero equilibria?
print("\nHow many states have non-zero equilibria?")
non_zero_eq = 0
for s in states:
    for eq in [approx.cce[s][a][0] for a in approx.cce[s]]:
        if np.sum(eq) > 0:
            non_zero_eq += 1
print("Number of states with non-zero equilibria: ", non_zero_eq)

# Average equilibrium value
print("\nOut of all cce values, what is the average equilibrium value?")
average_eq = 0
num = 0
for s in states:
    for eq in [approx.cce[s][a][0] for a in approx.cce[s]]:
        average_eq += np.sum(eq)
        num += 1
average_eq = average_eq / num
print("Average equilibrium value: ", average_eq)

# Average equilibrium value for states with over 5000 visits
average_eq1, average_eq2, average_eq3 = 0, 0, 0
num1, num2, num3 = 0, 0, 0
balance1 = 100
balance2 = 5000
balance3 = 25000
print(f"\nWhat is the average equilibrium value for states with over {balance1} visits? What about {balance2} visits? What about {balance3} visits?")
for s in states:
    for a in approx.cce[s]:
        if approx.cce[s][a][1] > balance1:
            for eq in [approx.cce[s][a][0] for a in approx.cce[s]]:
                average_eq1 += np.sum(eq)
                num1 += 1
        if approx.cce[s][a][1] > balance2:
            for eq in [approx.cce[s][a][0] for a in approx.cce[s]]:
                average_eq2 += np.sum(eq)
                num2 += 1
        if approx.cce[s][a][1] > balance3:
            for eq in [approx.cce[s][a][0] for a in approx.cce[s]]:
                average_eq3 += np.sum(eq)
                num3 += 1
average_eq1 = average_eq1 / num1
average_eq2 = average_eq2 / num2
average_eq3 = average_eq3 / num3
print(f"Average equilibrium value for states with over {balance1} visits: ", average_eq1)
print(f"Average equilibrium value for states with over {balance2} visits: ", average_eq2)
print(f"Average equilibrium value for states with over {balance3} visits: ", average_eq3)

# what is the hightest equilibrium value?
print("\nWhat is the highest equilibrium value?")
highest_eq = 0
for s in states:
    for eq in [approx.cce[s][a][0] for a in approx.cce[s]]:
        if np.max(eq) > highest_eq:
            highest_eq = np.max(eq)
print("Highest equilibrium value: ", highest_eq)

# what is the lowest equilibrium value?
print("\nWhat is the lowest equilibrium value?")
lowest_eq = 100000
for s in states:
    for eq in [approx.cce[s][a][0] for a in approx.cce[s]]:
        if np.min(eq) < lowest_eq:
            lowest_eq = np.min(eq)
print("Lowest equilibrium value: ", lowest_eq)

# what is the highest, lowest, and average visit counts for each state?
print("\nWhat is the highest, lowest, and average visit counts for each state?")
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

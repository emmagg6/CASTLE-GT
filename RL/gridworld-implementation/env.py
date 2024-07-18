import numpy as np

class CircleGridWorld:
    def __init__(self, size, radius, goal_state=None):
        self.size = size
        self.radius = radius
        self.agent_pos = [size // 2, size // 2]  # start at the center
        if goal_state is None:
            self.goal_pos = [13, 7]
            print('Goal state None is set to default')
            goal_state = self.goal_pos
        if self.check_goal_state(goal_state) is False:
            self.goal_pos = [13, 7]
        self.goal_pos = goal_state

    def reset(self):
        self.agent_pos = [self.size // 2, self.size // 2]
        return self.agent_pos

    def step(self, action):
        # action is a tuple of (-1, 0, 1) values for x and y direction
        new_pos = [self.agent_pos[0] + action[0], self.agent_pos[1] + action[1]]

        # Check if new position is within the circle
        distance_from_center = np.sqrt((new_pos[0] - self.size // 2) ** 2 + 
                                       (new_pos[1] - self.size // 2) ** 2)
        if distance_from_center + 0.5 <= self.radius:
            self.agent_pos = new_pos

        # Check if agent has reached the goal
        if self.agent_pos == self.goal_pos:
            reward = 10
            done = True
        else:
            reward = -0.01
            done = False

        return self.agent_pos, reward, done
    
    # check goal state is in the circle so it can be reached
    def check_goal_state(self, potential_goal):
        distance_from_center = np.sqrt((potential_goal[0] - self.size // 2) ** 2 + 
                                       (potential_goal[1] - self.size // 2) ** 2)
        if distance_from_center + 0.5 <= self.radius:
            print('Goal state is reachable')
            return True
        else:
            print('Goal state is not reachable')
            return False
    
    

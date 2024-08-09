import pandas as pd
import numpy as np

class GridWorld:
    def __init__(self):
        self.size = 5
        self.win_status = [0,4]
        self.lose_status = [0,2]
        self.block_status = [[2,2], [1,2]]
        self.initial_state = [4,0]

    def get_reward(self, state):
        if state == self.win_status:
            return 1
        elif state == self.lose_status:
            return -1
        else:
            return 0

    def next_state(self, current_state, movement):
        new_state = current_state[:]
        if movement == 'up':
            new_state[0] -= 1
        elif movement == 'down':
            new_state[0] += 1
        elif movement == 'left':
            new_state[1] -= 1
        elif movement == 'right':
            new_state[1] += 1

        if new_state in self.block_status:
            new_state = current_state

        new_state[0] = max(0, min(new_state[0], self.size - 1))
        new_state[1] = max(0, min(new_state[1], self.size - 1))

        return new_state

    def Is_terminal(self, state):
        if (state == self.win_status) or (state == self.lose_status):
            return True
        else:
            return False

    def Is_block(self, state):
        return (state in self.block_status)


class Policy:

    def __init__(self):
        self.grid_world = GridWorld()
        self.V = pd.DataFrame(np.zeros((self.grid_world.size, self.grid_world.size)))
        self.V_New = pd.DataFrame(np.zeros((self.grid_world.size, self.grid_world.size)))

        self.theta = 0.001
        self.gamma = 0.9
        self.epsilon = 0.3
        self.action_range = ['up', 'down', 'left', 'right']

    def action_selector(self):
        rand = np.random.rand()
        if rand < 0.25:
            action = 'up'
        elif rand < 0.5:
            action = 'down'
        elif rand < 0.75:
            action = 'left'
        else:
            action = 'right'

        return action

    def greedy_action_selector(self, current_state):
        rand = np.random.rand()
        if rand > self.epsilon:
            max = -float('inf')
            for movement in self.action_range:
                next_state = self.grid_world.next_state(current_state, movement)
                if self.V.iloc[next_state[0], next_state[1]] > max:
                    action = movement
                    max = self.V.iloc[next_state[0], next_state[1]]
            return action
        else:
            action = self.action_selector()
            return action

    def V_updator_random(self, current_state):

        if self.grid_world.Is_terminal(current_state):
            return self.grid_world.get_reward(current_state)
        elif self.grid_world.Is_block(current_state):
            return 0
        else:
            term = 0

            for action in self.action_range:
                next_state = self.grid_world.next_state(current_state, action)
                V_sprime = self.V.iloc[next_state[0], next_state[1]]
                r = self.grid_world.get_reward(next_state)
                term += (1 / len(self.action_range)) * (r + self.gamma * V_sprime)

            return term

    def V_updator_greedy(self, current_state, action):

        if self.grid_world.Is_terminal(current_state):
            return self.grid_world.get_reward(current_state)
        elif self.grid_world.Is_block(current_state):
            return 0
        else:
            term = 0

            next_state = self.grid_world.next_state(current_state, action)
            V_sprime = self.V.iloc[next_state[0], next_state[1]]
            r = self.grid_world.get_reward(next_state)
            term += (r + self.gamma * V_sprime)

            return term

delta = float('inf')

policy = Policy()

iteration = 0

while delta > policy.theta:
    delta = 0
    V_new = policy.V.copy()
    for i in range(policy.grid_world.size):
        for j in range(policy.grid_world.size):
            state = [i, j]
            V_prime = policy.V_updator_random(state)
            delta = max(delta, np.abs(V_prime - policy.V.iloc[state[0], state[1]]))
            V_new.iloc[state[0], state[1]] = V_prime
    policy.V = V_new
    iteration += 1
    print(f"Iteration {iteration}: Delta = {delta}")


policy = Policy()

policy_evaluation_iterations = 100
for _ in range(policy_evaluation_iterations):
    state = policy.grid_world.initial_state
    while not policy.grid_world.Is_terminal(state):
        action = policy.greedy_action_selector(state)
        next_state = policy.grid_world.next_state(state, action)
        V_prime = policy.V_updator_greedy(state, action)
        policy.V.iloc[state[0], state[1]] = V_prime
        state = next_state

    print(f"Iteration {_}")

V_new
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Times New Roman'
plt.figure(figsize = (10,8))
plt.title("Grid World")
plt.rcParams.update({'font.size': 12})
sns.heatmap(V_new, cmap = 'Spectral', vmin = -1, vmax = 1, center = 0, annot=True, fmt=".2f", square=True, linewidths=.5)
plt.show()

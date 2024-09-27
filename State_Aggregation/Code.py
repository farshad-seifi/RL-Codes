import numpy as np
import pandas as pd
from random import random
import matplotlib.pyplot as plt

class RandomWalk:
    def __init__(self):
        self.StartPoint = 500
        self.FinalPoint = 1000
        self.BadPoint = 0
        self.alpha = 2 * 10**-5
        self.gamma = 1
        self.current_state = 500
        self.is_finish = 0

    def action_selector(self):
        rand = random()
        if rand >= 0.5:
            action = +1
        else:
            action = -1

        return action

    def state_updator(self, action):
        new_state = self.current_state + action
        return new_state

    def check_finish(self):
        if self.current_state == self.FinalPoint:
            self.is_finish = 1
            reward = 1
            self.current_state = self.StartPoint
        elif self.current_state == self.BadPoint:
            self.is_finish = 1
            reward = -1
            self.current_state = self.StartPoint

        else:
            self.is_finish = 0
            reward = 0

        return reward

w = pd.DataFrame(np.zeros((10,1)))
state_values = pd.DataFrame(np.zeros((10,3)))
R = RandomWalk()
num_sim = 250
for j in range(0, num_sim):
    visited_states = [R.current_state]
    R.is_finish = 0
    while not R.is_finish:
        action = R.action_selector()
        R.current_state = R.state_updator(action)
        visited_states.append(R.current_state)
        Reward = R.check_finish()

    for i in range(0, len(visited_states) - 1):
        group = visited_states[i] // 100
        state_values[0].iloc[group] += 1
        state_values[1].iloc[group] += Reward
        state_values[2].iloc[group] = state_values[1].iloc[group] / state_values[0].iloc[group]
        w[0].iloc[group] = w[0].iloc[group] + R.alpha * (state_values[2].iloc[group] - w[0].iloc[group])

    print(j)



# Given array
w = np.array(w)

# Create the corresponding x-values (ranges of 100: 0-100, 101-200, ..., 901-1000)
x_values = np.arange(0, 1001, 100)  # Range from 0 to 1000

# Create step-like y-values
y_values = np.repeat(w, 2)  # Repeat each value twice to create the steps
x_steps = np.repeat(x_values[:-1], 2)  # Repeat x-values except for the last one

# Append the last point to complete the steps
x_steps = np.append(x_steps, 1000)
y_values = np.append(y_values, w[-1])

# Plotting
plt.rcParams['font.family'] = 'Times New Roman'

plt.figure(figsize=(8, 6))
plt.step(x_steps, y_values, where='post', linestyle='-', color='b', label='Estimated value')

# Add a line from (0, -1) to (1000, 1)
plt.plot([0, 1000], [-1, 1], linestyle='--', color='r', label='True value')

# Title and labels
plt.title('Estimated values using state aggregation')
plt.xlabel('State')
plt.ylabel('Estimated value')
plt.grid(True)
plt.legend()
# plt.savefig('State_Aggregation.png', dpi= 1200)

# Show the plot
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class windy_world:
    def __init__(self):
        self.columns_number = 10
        self.rows_number = 7
        self.start_point = [0, 3]
        self.final_point = [7, 3]
        self.wind_power = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
        self.epsilon = 0.1
        self.alpha = 0.5
        self.gamma = 1
        self.current_state = self.start_point
        self.Q_values = np.zeros((4, self.columns_number, self.rows_number))
        self.actions_list = ['u', 'd', 'r', 'l']

    def select_action(self):
        rand = np.random.rand()
        if rand <= self.epsilon:
            action = np.random.choice(self.actions_list)
        else:
            x = self.current_state[0]
            y = self.current_state[1]
            l = np.array([self.Q_values[0][x][y], self.Q_values[1][x][y], self.Q_values[2][x][y], self.Q_values[3][x][y]])
            action = self.actions_list[l.argmax()]
        return action

    def next_state(self, action):
        x = self.current_state[0]
        y = self.current_state[1]
        if action == 'u':
            y -= 1
        elif action == 'd':
            y += 1
        elif action == 'r':
            x += 1
        elif action == 'l':
            x -= 1

        y -= self.wind_power[min(x, self.columns_number - 1)]

        y = max(y, 0)
        y = min(y, self.rows_number - 1)
        x = max(0, x)
        x = min(x,self.columns_number - 1)

        return [x, y]

    def reward_calculator(self, next_state):
        if next_state == self.final_point:
            reward = 1
            is_terminate_state = True
        else:
            reward = -1
            is_terminate_state = False

        return reward, is_terminate_state

    def reset_game(self):
        self.current_state = self.start_point

    def Q_value_updator(self, current_state, action, reward, next_state, next_action):
        action_num = self.actions_list.index(action)
        next_action_num = self.actions_list.index(next_action)
        x = current_state[0]
        y = current_state[1]
        next_x = next_state[0]
        next_y = next_state[1]
        self.Q_values[action_num][x][y] = self.Q_values[action_num][x][y] +\
                                          self.alpha * (reward + self.gamma * self.Q_values[next_action_num][next_x][next_y] - self.Q_values[action_num][x][y])

simulations = 10000
episode_count = pd.DataFrame(np.zeros((simulations,1)))
w = windy_world()

for i in range(0, simulations):
    current_state = w.current_state
    action = w.select_action()
    next_state = w.next_state(action)
    reward, is_terminate_state = w.reward_calculator(next_state)
    if is_terminate_state:
        w.reset_game()
        episode_count[0].iloc[i] = episode_count[0].iloc[i-1] + 1
        w.Q_values[w.actions_list.index(action)][current_state[0]][current_state[1]] = reward * w.gamma
    else:
        w.current_state = next_state
        next_action = w.select_action()
        w.Q_value_updator(current_state, action, reward, next_state, next_action)
        if i >= 1:
            episode_count[0].iloc[i] = episode_count[0].iloc[i-1]




plt.rcParams['font.family'] = 'Times New Roman'
plt.figure(figsize=(10, 6))
plt.plot(episode_count.index, episode_count[0], color='blue')


# Adding labels and title
plt.xlabel('Time Steps')
plt.ylabel('Number of episodes')
plt.title('Results of Sarsa applied to the windy gridworld')

# Adding a legend to differentiate the lines
plt.legend()

# plt.savefig('Sarsa.png', dpi= 1200)
# Displaying the plot
plt.show()

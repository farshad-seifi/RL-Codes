import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
pd.options.mode.chained_assignment = None  # Suppress SettingWithCopyWarning

class Maze:
    def __init__(self):
        self.rows = 6
        self.columns = 9
        self.start_point = [0, 5]
        self.end_point = [8, 0]
        self.blocks = [[1, 2], [2, 2], [3, 2], [4, 5], [0, 7], [1, 7], [2, 7]]
        self.actions_list = ['u', 'd', 'r', 'l']
        self.epsilon = 0.1
        self.alpha = 0.5
        self.gamma = 0.95
        self.current_state = self.start_point
        self.Q_values = np.zeros((4, self.columns, self.rows))
        self.Model = {}

    def select_action(self):
        rand = np.random.rand()
        if rand <= self.epsilon:
            action = np.random.choice(self.actions_list)
        else:
            x = self.current_state[0]
            y = self.current_state[1]
            l = np.array([self.Q_values[0][x][y], self.Q_values[1][x][y], self.Q_values[2][x][y], self.Q_values[3][x][y]])

            max_value = np.max(l)
            max_indices = np.where(l == max_value)[0]
            selected_index = random.choice(max_indices)

            action = self.actions_list[selected_index]

        return action

    def select_action_max(self):
        x = self.current_state[0]
        y = self.current_state[1]
        l = np.array([self.Q_values[0][x][y], self.Q_values[1][x][y], self.Q_values[2][x][y], self.Q_values[3][x][y]])

        max_value = np.max(l)
        max_indices = np.where(l == max_value)[0]
        selected_index = random.choice(max_indices)

        action = self.actions_list[selected_index]


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

        if [x, y] in self.blocks:
            x = self.current_state[0]
            y = self.current_state[1]
        else:
            y = max(y, 0)
            y = min(y, self.rows - 1)
            x = max(0, x)
            x = min(x,self.columns - 1)

        return [x, y]

    def reward_calculator(self, next_state):

        if next_state == self.end_point:
            reward = 1
            is_terminate_state = True
        else:
            reward = 0
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

    def Model_updator(self, current_state, action, reward, next_state):

        x = current_state[0]
        y = current_state[1]
        x_prime = next_state[0]
        y_prime = next_state[1]

        if (x, y) not in self.Model:
            # If not, add the state with an empty dictionary as its value
            self.Model[(x, y)] = {}

        if action not in self.Model[(x, y)]:
            # If not, add the action with its corresponding next_state and reward
            self.Model[(x, y)][action] = ((x_prime, y_prime), reward)

    def planner_selector(self):
        states = list(self.Model.keys())
        selected_state = np.random.choice(len(states))
        selected_state = states[selected_state]

        acts = list(self.Model[selected_state].keys())
        selected_action = np.random.choice(len(acts))
        selected_action = acts[selected_action]

        next_state = self.Model[selected_state][selected_action][0]
        next_reward = self.Model[selected_state][selected_action][1]

        return selected_state, selected_action, next_state, next_reward

simulations = 10000
w = Maze()
planning_steps = 100
Total_Reward = 0
Reward_Q_Learning = pd.DataFrame(np.zeros((500, 10)))
steps = 0
steps_counter = pd.DataFrame(np.zeros((simulations, 2)))

for i in range(0, simulations):
    current_state = w.current_state
    action = w.select_action()
    next_state = w.next_state(action)
    reward, is_terminate_state = w.reward_calculator(next_state)

    if is_terminate_state:

        w.Q_values[w.actions_list.index(action)][current_state[0]][current_state[1]] = reward * w.gamma

        for j in range(0, planning_steps):
            selected_state, selected_action, next_state, next_reward = w.planner_selector()
            w.current_state = next_state
            next_action = w.select_action_max()
            w.Q_value_updator(selected_state, selected_action, next_reward, next_state, next_action)

        w.reset_game()
        steps = 0
    else:
        w.current_state = next_state
        next_action = w.select_action_max()
        w.Q_value_updator(current_state, action, reward, next_state, next_action)
        w.Model_updator(current_state, action, reward, next_state)
        steps += 1

    steps_counter[0].iloc[i] = steps
    steps_counter[1].iloc[i] = is_terminate_state

a = steps_counter[steps_counter[1] == True].index
final_steps = steps_counter[0].iloc[a-1]
final_steps = pd.DataFrame(final_steps)
final_steps.reset_index(inplace= True, drop= True)

a = steps_counter[steps_counter[1] == True].index
final_steps_0 = steps_counter[0].iloc[a-1]
final_steps_0 = pd.DataFrame(final_steps_0)
final_steps_0.reset_index(inplace= True, drop= True)




plt.rcParams['font.family'] = 'Times New Roman'
plt.figure(figsize=(10, 6))
plt.plot(final_steps.iloc[2:50].index, final_steps.iloc[2:50][0], color='green', label="DynaQ")
plt.plot(final_steps_0.iloc[2:50].index, final_steps_0.iloc[2:50][0], color='blue', label="Q-Learning")


# Adding labels and title
plt.xlabel('Episode')
plt.ylabel('Steps per Episode')
plt.title('Results of DynaQ andQ-Learning applied to the Maze problem')

# Adding a legend to differentiate the lines
plt.legend()

# plt.savefig('Maze.png', dpi= 1200)
# Displaying the plot
plt.show()

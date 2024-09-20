import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
pd.options.mode.chained_assignment = None  # Suppress SettingWithCopyWarning

class Maze:
    def __init__(self):
        self.rows = 6
        self.columns = 9
        self.start_point = [3, 5]
        self.end_point = [8, 0]
        self.blocks = [[0,3], [1,3], [2,3], [3,3], [4,3], [5,3], [6,3], [7,3]]
        self.actions_list = ['u', 'd', 'r', 'l']
        self.epsilon = 0.1
        self.alpha = 0.5
        self.gamma = 0.95
        self.kapa = 0
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

    def get_model_value(self, x, y, action):
        # Check if (x, y) is in the model dictionary
        if (x, y) in self.Model:
            # Check if action exists in the model for (x, y)
            if action in self.Model[(x, y)]:
                # Access the third element in the tuple (if it exists)
                return self.Model[(x, y)][action][2]

        # Return 0 if either the state-action pair or the third element doesn't exist
        return 0

    def select_action_max(self, time):
        x = self.current_state[0]
        y = self.current_state[1]

        value = [0,0,0,0]
        for k in range(0,4):
            action = self.actions_list[k]
            last_time = self.get_model_value(x,y,action)
            value[k] = self.kapa * np.sqrt(time - last_time)

        l1 = np.array([self.Q_values[0][x][y] , self.Q_values[1][x][y] , self.Q_values[2][x][y] , self.Q_values[3][x][y]] )
        l = l1 + value
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

    def Model_updator(self, current_state, action, reward, next_state, time):

        x = current_state[0]
        y = current_state[1]
        x_prime = next_state[0]
        y_prime = next_state[1]

        if (x, y) not in self.Model:
            # If not, add the state with an empty dictionary as its value
            self.Model[(x, y)] = {}

        self.Model[(x, y)][action] = ((x_prime, y_prime), reward, time)

    def planner_selector(self, time):
        states = list(self.Model.keys())
        selected_state = np.random.choice(len(states))
        selected_state = states[selected_state]

        acts = list(self.Model[selected_state].keys())
        selected_action = np.random.choice(len(acts))
        selected_action = acts[selected_action]

        next_state = self.Model[selected_state][selected_action][0]
        last_time = self.Model[selected_state][selected_action][2]
        # next_reward = self.Model[selected_state][selected_action][1] + self.kapa * np.sqrt(time - last_time)
        next_reward = self.Model[selected_state][selected_action][1]

        return selected_state, selected_action, next_state, next_reward

simulations = 10000
w = Maze()
planning_steps = 1000
Total_Reward = 0
Reward_Q_Learning = pd.DataFrame(np.zeros((500, 10)))
steps = 0
Cumulative_Reward = pd.DataFrame(np.zeros((simulations, 2)))

for i in range(0, simulations):
    if i == 3000:
        w.blocks = [[1,3], [2,3], [3,3], [4,3], [5,3], [6,3], [7,3], [8,3]]
    current_state = w.current_state
    if i == 0:
        action = w.select_action()
    else:
        rand = np.random.rand()
        if rand > w.epsilon:
            action = w.select_action_max(i)
        else:
            action = np.random.choice(w.actions_list)

    next_state = w.next_state(action)
    reward, is_terminate_state = w.reward_calculator(next_state)

    if is_terminate_state:

        w.Q_values[w.actions_list.index(action)][current_state[0]][current_state[1]] = reward * w.gamma

        for j in range(0, planning_steps):
            selected_state, selected_action, next_state, next_reward = w.planner_selector(i)
            w.current_state = next_state
            next_action = w.select_action_max(i)
            w.Q_value_updator(selected_state, selected_action, next_reward, next_state, next_action)

        w.reset_game()
        steps = 0
    else:
        w.current_state = next_state
        next_action = w.select_action_max(i)
        w.Q_value_updator(current_state, action, reward, next_state, next_action)
        w.Model_updator(current_state, action, reward, next_state,i)
        steps += 1
    Total_Reward += reward

    Cumulative_Reward[0].iloc[i] = reward
    Cumulative_Reward[1].iloc[i] = Total_Reward


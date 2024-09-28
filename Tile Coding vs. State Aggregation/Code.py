import numpy as np
import pandas as pd
from random import random
import matplotlib.pyplot as plt

class RandomWalk:
    def __init__(self):
        self.StartPoint = 500
        self.FinalPoint = 1000
        self.BadPoint = 0
        self.alpha = 0.01
        self.gamma = 1
        self.current_state = 500
        self.is_finish = 0

    def action_selector(self):
        rand = random()
        if rand >= 0.5:
            action = + np.round(random() * 200)
        else:
            action = - np.round(random() * 200)

        return action

    def state_updator(self, action):
        new_state = self.current_state + action
        return new_state

    def check_finish(self):
        if self.current_state >= self.FinalPoint:
            self.is_finish = 1
            reward = 1
            self.current_state = self.StartPoint
        elif self.current_state <= self.BadPoint:
            self.is_finish = 1
            reward = -1
            self.current_state = self.StartPoint

        else:
            self.is_finish = 0
            reward = 0

        return reward

for p in range(0,5):
    episodes = 3000
    w = pd.DataFrame(np.zeros((5, 1)))
    R = RandomWalk()
    R.alpha = 0.0001

    VE = pd.DataFrame(np.zeros((episodes, 5)))
    for j in range(0, episodes):
        R.is_finish = 0
        while not R.is_finish:
            action = R.action_selector()
            first_state = R.current_state

            R.current_state = R.state_updator(action)
            next_state = R.current_state
            Reward = R.check_finish()
            first_w = int(first_state / 200)
            next_w = int(next_state / 200)
            if next_state >= 1000:
                next_w = 4
            w.iloc[first_w] = w.iloc[first_w] + R.alpha * (Reward + R.gamma * w.iloc[next_w] - w.iloc[first_w])

        ve_value = 0
        for k in range(0, 1000):
            loc = int(k / 200)
            ve_value += (w.iloc[loc] - ((2 / 1000) * k - 1)) ** 2

        VE[p].iloc[j] = ve_value
        print(p,j)

    tile = pd.DataFrame(np.zeros((300, 1)))
    R = RandomWalk()
    R.alpha = 0.0001 / 50
    VE_Tile = pd.DataFrame(np.zeros((episodes, 5)))
    for j in range(0, episodes):
        R.is_finish = 0
        while not R.is_finish:
            action = R.action_selector()
            first_state = R.current_state

            R.current_state = R.state_updator(action)
            next_state = R.current_state
            Reward = R.check_finish()
            first_w_value = 0
            next_w_value = 0
            for i in range(0, 50):
                first_w = int((first_state + 200 - i * 4) / 200) + 6 * i
                first_w_value += tile.iloc[first_w]
                if next_state >= 1000:
                    next_state = 999
                next_w = int((next_state + 200 - i * 4) / 200) + 6 * i
                next_w_value += tile.iloc[next_w]

                tile.iloc[first_w] = tile.iloc[first_w] + R.alpha * (Reward + R.gamma * next_w_value - first_w_value)

        ve_value = 0
        for k in range(0, 1000):
            first_w_value = 0
            for o in range(0, 50):
                first_w = int((k + 200 - o * 4) / 200) + 6 * o
                first_w_value += tile.iloc[first_w]

            ve_value += (first_w_value - ((2 / 1000) * k - 1)) ** 2

        VE_Tile[p].iloc[j] = ve_value
        print(p,j)

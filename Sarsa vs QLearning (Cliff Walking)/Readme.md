In the cliff walking task we are encountered with a cliff and reaching it results -100 reward and reset the game. As it has shown in the following picture, we have two options. The safer path and the optimal one.
![cliffwalk](https://github.com/user-attachments/assets/8002ebed-4ce1-418e-9fbc-cc1778c0c0e3)

Due to the fact that Q-Learning updates action values using the optimal policy, it tends to follow the optimal path, while following th behavoiur policy (epsilon greedy) can pose more negative reward to it compared to safer path (due to the exploration). However, the Sarsa method follows safer path because it is an online policy method and updates and follows epsilon greedy policy simultaneously. It has led to better average reward compared to Q-Learning approach.

![Cliff](https://github.com/user-attachments/assets/76f2d17f-6504-41b5-9d3f-2482fc261d95)

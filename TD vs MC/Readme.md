Due to the fact that TD methods bootstraps during learning, they can adjust to real state values faster compared to MC ones. To check this, here a random walk framework with +1 reward in state 3 and -1 in state -3 has been investigated.
Firstly by solving Bellman equation, actual state values has been calculated and then using these algorithms the values has been estimated. As shown in the following plot,TD methods converge faster than MC ones.
![TD vs MC](https://github.com/user-attachments/assets/c8b9ddc8-c27f-4a7f-845d-a3cfb5f5c8db)

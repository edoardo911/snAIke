# snAIke

snAIke is a DQN neural network that learns how to play snake in a 9x9 map.
It is written in python using pytorch and uses a reinforcement learning model based
on rewards to understand the rules of the game and maximize the score.
snAIke implements a epsilon-greedy algorithm to allow an exploration phase for the first attempts (~950).
The network has 90 inputs (one for each tile, one for the snake direction and 8 for every tile around the snake),
a first hidden layer with 256 neurons, a second hidden layer with 128 neurons and 4 outputs (up, down, bottom, left)

Install requirements:
`pip install -r requirements.txt`
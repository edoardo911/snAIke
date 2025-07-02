import os
import time
from random import randrange
import torch
from neural_network.DQN import DQN
from neural_network.ReplayBuffer import ReplayBuffer
from neural_network.DQN import select_action
from neural_network.DQN import train
import torch.optim as optim
import copy

ID_AIR: int = 0
ID_WALL: int = 1
ID_TAIL: int = 2
ID_FOOD: int = 3
ID_HEAD: int = 4

DIR_UP: int = 0
DIR_RIGHT: int = 1
DIR_DOWN: int = 2
DIR_LEFT: int = 3

game_map = []
snake_pos: list = []
tail: list = []
snake_dir: int = -1
attempt: int = 1
record: int = 0
game_over: bool = False

print("Loading...")

#dqn
model = DQN()
target_model = DQN()

target_model.load_state_dict(model.state_dict())
target_model.eval()

memory = ReplayBuffer()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

training_count = 0
copy_count = 0

epsilon = 1.0
epsilon_decay = 0.995

def resetState():
    global game_map
    global snake_pos
    global game_over

    game_over = False
    snake_pos = [ randrange(7) + 1, randrange(7) + 1 ]
    game_map = [ #9x9
        [ 1, 1, 1, 1, 1, 1, 1, 1, 1 ],
        [ 1, 0, 0, 0, 0, 0, 0, 0, 1 ],
        [ 1, 0, 0, 0, 0, 0, 0, 0, 1 ],
        [ 1, 0, 0, 0, 0, 0, 0, 0, 1 ],
        [ 1, 0, 0, 0, 0, 0, 0, 0, 1 ],
        [ 1, 0, 0, 0, 0, 0, 0, 0, 1 ],
        [ 1, 0, 0, 0, 0, 0, 0, 0, 1 ],
        [ 1, 0, 0, 0, 0, 0, 0, 0, 1 ],
        [ 1, 1, 1, 1, 1, 1, 1, 1, 1 ],
    ]
    game_map[snake_pos[1]][snake_pos[0]] = ID_HEAD

def idToChar(id: int):
    if id == ID_AIR:
        return ' '
    elif id == ID_WALL:
        return '#'
    elif id == ID_TAIL:
        return 'o'
    elif id == ID_FOOD:
        return 'X'
    elif id == ID_HEAD:
        return 'O'
    
def moveSnake():
    global snake_pos
    global game_map
    game_map[snake_pos[1]][snake_pos[0]] = ID_AIR
    x = snake_pos[0]
    y = snake_pos[1]
    reward = 0

    if snake_dir == DIR_UP:
        y -= 1
    elif snake_dir == DIR_RIGHT:
        x += 1
    elif snake_dir == DIR_DOWN:
        y += 1
    elif snake_dir == DIR_LEFT:
        x -= 1
    if checkGameOver(x, y):
        global game_over
        game_over = True
        reward = -1
    else:
        for t in tail:
            game_map[t[1]][t[0]] = ID_AIR

        if checkFood(x, y):
            reward = 1
            tail.append([ snake_pos[0], snake_pos[1] ])
        elif len(tail) > 0:
            for i in range(len(tail) - 1):
                tail[i] = tail[i + 1]
            tail[-1] = [ snake_pos[0], snake_pos[1] ]
        for t in tail:
            game_map[t[1]][t[0]] = ID_TAIL
        snake_pos = [ x, y ]
        game_map[snake_pos[1]][snake_pos[0]] = ID_HEAD
    return reward

def checkFood(x: int, y: int):
    id = game_map[y][x]
    if id == ID_FOOD:
        return True
    return False

def checkGameOver(x: int, y: int):
    id = game_map[y][x]
    if id != ID_FOOD and id != ID_AIR:
        return True
    return False

def generateFood():
    global game_map
    x = 0
    y = 0

    chance = randrange(5)
    if chance != 0:
        return
    while game_map[y][x] != ID_AIR:
        x = randrange(7) + 1
        y = randrange(7) + 1
    game_map[y][x] = ID_FOOD

resetState()
os.system("cls")
while True:
    x: int = 0
    y: int = 0

    print("Attempt: " + str(attempt) + ", Score: " + str(len(tail)) + ", Record: " + str(record) + ", Epsilon: " + str(epsilon))
    for row in game_map:
        for id in row:
            print(idToChar(id) + ' ', end='',  sep='')
            x += 1
        print()
        y += 1

    unnormalized_state = copy.deepcopy(game_map)
    flat_state = [ item for row in unnormalized_state for item in row ]
    normalized_state = [x / 4 for x in flat_state]

    action = select_action(model, normalized_state, epsilon)
    snake_dir = action

    epsilon = max(epsilon * epsilon_decay, 0.01)
    
    reward = moveSnake()
    generateFood()

    new_unnormalized_state = copy.deepcopy(game_map)
    new_flat_state = [ item for row in new_unnormalized_state for item in row ]
    new_normalized_state = [x / 4 for x in new_flat_state]
    memory.push(normalized_state, action, reward, new_normalized_state, game_over)

    if training_count == 3 or game_over:
        training_count = 0
        train(model, target_model, memory, optimizer, 0.99, 32)
    else:
        training_count += 1

    if copy_count == 1000:
        copy_count = 0
        target_model.load_state_dict(model.state_dict())
    else:
        copy_count += 1
    
    time.sleep(0.05)
    os.system("cls")

    if game_over:
        attempt += 1
        if len(tail) > record:
            record = len(tail)
        tail.clear()
        print("Game Over!")
        time.sleep(1)
        resetState()
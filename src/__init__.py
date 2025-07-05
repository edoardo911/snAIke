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
import pygame
import pickle

ID_AIR: int = 0
ID_WALL: int = 1
ID_TAIL: int = 2
ID_FOOD: int = 3
ID_HEAD: int = 4

DIR_UP: int = 0
DIR_RIGHT: int = 1
DIR_DOWN: int = 2
DIR_LEFT: int = 3

map_id: int = -1
game_map = []
snake_pos: list = [ 0, 0 ]
tail: list = []
snake_dir: int = -1
attempt: int = 0
record: int = 0
game_over: bool = False
score_sum: float = 0

die_reward: float = -1
eat_reward: float = 1
survive_reward: float = 0.02
survive_decay: float = 0.9

print("Loading...")

with open("snake.conf", "r") as f:
    lines = [line.rstrip() for line in f]
    for line in lines:
        tokens = line.split("=")
        if tokens[0] == "map_id":
            map_id = int(tokens[1])
        elif tokens[0] == "die_reward":
            die_reward = float(tokens[1])
        elif tokens[0] == "eat_reward":
            eat_reward = float(tokens[1])
        elif tokens[0] == "survive_reward":
            survive_reward = float(tokens[1])
        elif tokens[0] == "survive_decay":
            survive_decay = float(tokens[1])

#dqn
model = DQN()
target_model = DQN()

epsilon = 1.0
epsilon_decay = 0.995

if os.path.isfile(f"model{map_id}.pth"):
    print("Loading model...")
    model.load_state_dict(torch.load(f"model{map_id}.pth"))
    with open(f"gamestate{map_id}.pkl", "rb") as f:
        data = pickle.load(f)
        record = data[0]
        attempt = data[1]
        score_sum = data[2]
        epsilon = data[3]
    print("Model loaded")

target_model.load_state_dict(model.state_dict())
target_model.eval()

memory = ReplayBuffer()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

training_count = 0
copy_count = 0
default_reward = survive_reward

#pygame
pygame.init()
screen = pygame.display.set_mode((400, 400))
pygame.display.set_caption("snAIke")
clock = pygame.time.Clock()

grass = pygame.image.load("res/grass.png").convert_alpha()
wall = pygame.image.load("res/wall.png").convert_alpha()
food = pygame.image.load("res/food.png").convert_alpha()
snake_tail = pygame.image.load("res/snake_tail.png").convert_alpha()
snake_head_down = pygame.image.load("res/snake_head.png").convert_alpha()
snake_head_right = pygame.transform.rotate(snake_head_down, 90)
snake_head_up = pygame.transform.rotate(snake_head_right, 90)
snake_head_left = pygame.transform.rotate(snake_head_up, 90)
font = pygame.font.SysFont("Arial", 14)

def resetState():
    global game_map
    global snake_pos
    global game_over
    global default_reward

    default_reward = survive_reward
    game_over = False
    if map_id == 0:
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
    elif map_id == 1:
        game_map = [ #9x9
            [ 1, 1, 1, 1, 1, 1, 1, 1, 1 ],
            [ 1, 0, 0, 0, 0, 0, 0, 0, 1 ],
            [ 1, 0, 0, 0, 0, 0, 0, 0, 1 ],
            [ 1, 0, 0, 0, 0, 0, 0, 0, 1 ],
            [ 1, 0, 0, 0, 1, 0, 0, 0, 1 ],
            [ 1, 0, 0, 0, 0, 0, 0, 0, 1 ],
            [ 1, 0, 0, 0, 0, 0, 0, 0, 1 ],
            [ 1, 0, 0, 0, 0, 0, 0, 0, 1 ],
            [ 1, 1, 1, 1, 1, 1, 1, 1, 1 ],
        ]
    elif map_id == 2:
        game_map = [ #9x9
            [ 1, 1, 1, 1, 1, 1, 1, 1, 1 ],
            [ 1, 0, 0, 0, 0, 0, 0, 0, 1 ],
            [ 1, 0, 1, 0, 0, 0, 1, 0, 1 ],
            [ 1, 0, 0, 0, 0, 0, 0, 0, 1 ],
            [ 1, 0, 0, 0, 0, 0, 0, 0, 1 ],
            [ 1, 0, 0, 0, 0, 0, 0, 0, 1 ],
            [ 1, 0, 1, 0, 0, 0, 1, 0, 1 ],
            [ 1, 0, 0, 0, 0, 0, 0, 0, 1 ],
            [ 1, 1, 1, 1, 1, 1, 1, 1, 1 ],
        ]
    elif map_id == 3:
        game_map = [ #9x9
            [ 1, 1, 1, 1, 1, 1, 1, 1, 1 ],
            [ 1, 1, 0, 0, 0, 0, 0, 1, 1 ],
            [ 1, 0, 0, 0, 0, 0, 0, 0, 1 ],
            [ 1, 0, 0, 0, 0, 0, 0, 0, 1 ],
            [ 1, 0, 0, 0, 1, 0, 0, 0, 1 ],
            [ 1, 0, 0, 0, 0, 0, 0, 0, 1 ],
            [ 1, 0, 0, 0, 0, 0, 0, 0, 1 ],
            [ 1, 1, 0, 0, 0, 0, 0, 1, 1 ],
            [ 1, 1, 1, 1, 1, 1, 1, 1, 1 ],
        ]
    elif map_id == 4:
        game_map = [ #9x9
            [ 1, 1, 1, 1, 1, 1, 1, 1, 1 ],
            [ 1, 0, 0, 0, 0, 0, 0, 0, 1 ],
            [ 1, 0, 0, 0, 1, 0, 0, 0, 1 ],
            [ 1, 0, 0, 1, 1, 1, 0, 0, 1 ],
            [ 1, 0, 1, 1, 1, 1, 1, 0, 1 ],
            [ 1, 0, 0, 1, 1, 1, 0, 0, 1 ],
            [ 1, 0, 0, 0, 1, 0, 0, 0, 1 ],
            [ 1, 0, 0, 0, 0, 0, 0, 0, 1 ],
            [ 1, 1, 1, 1, 1, 1, 1, 1, 1 ],
        ]
    else:
        print("Invalid map id!")
        exit()
    while game_map[snake_pos[1]][snake_pos[0]] != ID_AIR:
        snake_pos = [ randrange(7) + 1, randrange(7) + 1 ]
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
    global default_reward

    game_map[snake_pos[1]][snake_pos[0]] = ID_AIR
    x = snake_pos[0]
    y = snake_pos[1]
    reward = default_reward

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
        reward = die_reward
    else:
        for t in tail:
            game_map[t[1]][t[0]] = ID_AIR

        default_reward *= survive_decay
        if checkFood(x, y):
            reward = eat_reward
            default_reward = survive_reward
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

def drawMap():
    y = 4

    screen.fill((0, 0, 0))

    text = "Attempt: " + str(attempt) + ", Score: " + str(len(tail)) + ", Record: " + str(record) + ", Epsilon: " + str(round(epsilon, 3))
    surface = font.render(text, True, (255, 255, 255))
    screen.blit(surface, (5, 5))

    text = "Efficiency index: " + str(round(score_sum / attempt, 3)) if attempt != 0 else "0"
    surface = font.render(text, True, (255, 255, 255))
    screen.blit(surface, (5, 35))

    for row in game_map:
        x = 2
        for id in row:
            image = grass
            if id == ID_FOOD:
                screen.blit(image, (x * 16, y * 16))
                image = food
            elif id == ID_TAIL:
                screen.blit(image, (x * 16, y * 16))
                image = snake_tail
            elif id == ID_WALL:
                image = wall
            elif id == ID_HEAD:
                screen.blit(image, (x * 16, y * 16))
                if snake_dir == -1 or snake_dir == DIR_DOWN:
                    image = snake_head_down
                elif snake_dir == DIR_LEFT:
                    image = snake_head_left
                elif snake_dir == DIR_UP:
                    image = snake_head_up
                elif snake_dir == DIR_RIGHT:
                    image = snake_head_right
            screen.blit(image, (x * 16, y * 16))
            x += 1
        y += 1
    pygame.display.flip()
    clock.tick(10)

resetState()
os.system("cls")

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    drawMap()

    unnormalized_state = copy.deepcopy(game_map)
    flat_state = [ item for row in unnormalized_state for item in row ]
    normalized_state = [x / 4 for x in flat_state]
    normalized_state.append(snake_dir / 3)
    normalized_state.append(game_map[snake_pos[1] - 1][snake_pos[0] - 1] / 4)
    normalized_state.append(game_map[snake_pos[1] - 1][snake_pos[0]] / 4)
    normalized_state.append(game_map[snake_pos[1] - 1][snake_pos[0] + 1] / 4)
    normalized_state.append(game_map[snake_pos[1]][snake_pos[0] - 1] / 4)
    normalized_state.append(game_map[snake_pos[1]][snake_pos[0] + 1] / 4)
    normalized_state.append(game_map[snake_pos[1] + 1][snake_pos[0] - 1] / 4)
    normalized_state.append(game_map[snake_pos[1] + 1][snake_pos[0]] / 4)
    normalized_state.append(game_map[snake_pos[1] + 1][snake_pos[0] + 1] / 4)

    action = select_action(model, normalized_state, epsilon)
    snake_dir = action
    
    reward = moveSnake()
    generateFood()

    new_unnormalized_state = copy.deepcopy(game_map)
    new_flat_state = [ item for row in new_unnormalized_state for item in row ]
    new_normalized_state = [x / 4 for x in new_flat_state]
    new_normalized_state.append(snake_dir / 3)
    new_normalized_state.append(game_map[snake_pos[1] - 1][snake_pos[0] - 1] / 4)
    new_normalized_state.append(game_map[snake_pos[1] - 1][snake_pos[0]] / 4)
    new_normalized_state.append(game_map[snake_pos[1] - 1][snake_pos[0] + 1] / 4)
    new_normalized_state.append(game_map[snake_pos[1]][snake_pos[0] - 1] / 4)
    new_normalized_state.append(game_map[snake_pos[1]][snake_pos[0] + 1] / 4)
    new_normalized_state.append(game_map[snake_pos[1] + 1][snake_pos[0] - 1] / 4)
    new_normalized_state.append(game_map[snake_pos[1] + 1][snake_pos[0]] / 4)
    new_normalized_state.append(game_map[snake_pos[1] + 1][snake_pos[0] + 1] / 4)
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
    if game_over:
        epsilon = epsilon * epsilon_decay
        attempt += 1
        score_sum += len(tail)
        if len(tail) > record:
            record = len(tail)
        
        text = "Game Over! Score: " + str(len(tail))
        surface = font.render(text, True, (255, 0, 0))
        screen.blit(surface, (192, 120))
        pygame.display.flip()
        clock.tick(10)

        if attempt % 50 == 0:
            print("Saving model...")
            torch.save(model.state_dict(), f"model{map_id}.pth")
            with open(f"gamestate{map_id}.pkl", "wb") as f:
                pickle.dump((record, attempt, score_sum, epsilon), f)
            print("Model saved")

        tail.clear()
        time.sleep(1)
        resetState()
pygame.quit()
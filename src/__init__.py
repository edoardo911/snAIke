import os
import time
from random import randrange

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
game_over: bool = False

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
    else:
        for t in tail:
            game_map[t[1]][t[0]] = ID_AIR

        if checkFood(x, y):
            tail.append([ snake_pos[0], snake_pos[1] ])
        elif len(tail) > 0:
            for i in range(len(tail) - 1):
                tail[i] = tail[i + 1]
            tail[-1] = [ snake_pos[0], snake_pos[1] ]
        for t in tail:
            game_map[t[1]][t[0]] = ID_TAIL
        snake_pos = [ x, y ]
        game_map[snake_pos[1]][snake_pos[0]] = ID_HEAD

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

    chance = randrange(3)
    if chance != 0:
        return
    while game_map[y][x] != ID_AIR:
        x = randrange(7) + 1
        y = randrange(7) + 1
    game_map[y][x] = ID_FOOD

resetState()
while True:
    x: int = 0
    y: int = 0

    print("Attempt: " + str(attempt) + ", Score: " + str(len(tail)))
    for row in game_map:
        for id in row:
            print(idToChar(id) + ' ', end='',  sep='')
            x += 1
        print()
        y += 1

    #test
    char = input('input: ')
    if char == 'w':
        snake_dir = DIR_UP
    elif char == 'a':
        snake_dir = DIR_LEFT
    elif char == 's':
        snake_dir = DIR_DOWN
    elif char == 'd':
        snake_dir = DIR_RIGHT
    #test
    
    moveSnake()
    generateFood()
    
    time.sleep(0.3)
    os.system("cls")

    if game_over:
        attempt += 1
        print("Game Over!")
        input()
        resetState()
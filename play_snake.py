#!/usr/bin/env python3

from learn2slither.Snake import Snake

import pygame
import time

FPS = 60
MOVE_PER_SECOND = 5

if __name__ == "__main__":
    snake = Snake(render=True)
    counter = 0
    last_render = time.perf_counter()
    last_move = time.perf_counter()
    while True:
        now = time.perf_counter()
        if (now - last_move >= 1 / MOVE_PER_SECOND):
            counter += 1
            snake.move()
            counter += 1
            last_move = now
            reward = snake.get_reward()
            state = snake.get_state()
            print(f"reward = {reward}")
            print(f"state = {state}")
            print(f"counter = {counter}")
        if (now - last_render >= 1 / 60):
            snake.draw()
            snake.poll_input()
            last_render = now

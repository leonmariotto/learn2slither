#!/usr/bin/env python3

from snake.Snake import Snake

import pygame
import time

FPS = 60
MOVE_PER_SECOND = 5

if __name__ == "__main__":
    snake = Snake()
    last_render = time.perf_counter()
    last_move = time.perf_counter()
    while True:
        now = time.perf_counter()
        if (now - last_move >= 1 / MOVE_PER_SECOND):
            snake.poll_input()
            snake.move()
            last_move = now
        if (now - last_render >= 1 / 60):
            snake.draw()
            last_render = now

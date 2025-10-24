"""
Snake engine.
"""

import pygame
import sys
import random

# --- Game configuration ---
GRID_WIDTH, GRID_HEIGHT = 20, 20

# --- Rendering configuration ---
CELL_SIZE = 25
SCREEN_WIDTH, SCREEN_HEIGHT = CELL_SIZE * GRID_WIDTH, CELL_SIZE * GRID_HEIGHT

# Colors
COLORS = {
    "background": (30, 30, 30),
    "snake": (0, 200, 0),
    "food": (200, 0, 0),
    "grid": (40, 40, 40)
}

DIR_NAME = {(0,-1):"UP",(0,1):"DOWN",(-1,0):"LEFT",(1,0):"RIGHT"}

class Snake():
    def __init__(self, render:bool=False):
        # --- Initialize pygame ---
        self.render = render
        if self.render is True:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Learn2Slither")
        # --- Game state ---
        self.snake = [(5, 5), (4, 5), (3, 5)]
        self.direction = (1, 0)
        self.food = (10, 10)
        self.head_textures = {
            "UP": Snake.make_head_texture("UP"),
            "DOWN": Snake.make_head_texture("DOWN"),
            "LEFT": Snake.make_head_texture("LEFT"),
            "RIGHT": Snake.make_head_texture("RIGHT"),
        }


    def draw_grid(self):
        """Draw subtle grid lines."""
        if self.render is False:
            raise ValueError();
        for x in range(0, SCREEN_WIDTH, CELL_SIZE):
            pygame.draw.line(self.screen, COLORS["grid"], (x, 0), (x, SCREEN_HEIGHT))
        for y in range(0, SCREEN_HEIGHT, CELL_SIZE):
            pygame.draw.line(self.screen, COLORS["grid"], (0, y), (SCREEN_WIDTH, y))

    # Create directional head textures dynamically
    @staticmethod
    def make_head_texture(direction, color=(0,200,0)):
        surf = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
        c = CELL_SIZE // 2
        if direction == "UP":
            points = [(c, 0), (0, CELL_SIZE), (CELL_SIZE, CELL_SIZE)]
        elif direction == "DOWN":
            points = [(0, 0), (CELL_SIZE, 0), (c, CELL_SIZE)]
        elif direction == "LEFT":
            points = [(0, c), (CELL_SIZE, 0), (CELL_SIZE, CELL_SIZE)]
        elif direction == "RIGHT":
            points = [(0, 0), (CELL_SIZE, c), (0, CELL_SIZE)]
        else:
            raise ValueError()
        pygame.draw.polygon(surf, color, points)
        return surf

    def draw_snake(self):
        if self.render is False:
            raise ValueError();
        for i,(x,y) in enumerate(self.snake):
            rect = pygame.Rect(x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE)
            if i == 0:
                dir_name = DIR_NAME[self.direction]
                self.screen.blit(self.head_textures[dir_name], rect.topleft)
            elif i == len(self.snake)-1:
                pygame.draw.rect(self.screen, (0,150,0), rect, border_radius=10)
            else:
                pygame.draw.rect(self.screen, (0,180,0), rect, border_radius=4)
                pygame.draw.rect(self.screen, (0,100,0), rect, width=2, border_radius=4)

    def draw(self):
        if self.render is False:
            raise ValueError();
        """Draw the entire board based on known cell states."""
        self.screen.fill(COLORS["background"])
        self.draw_grid()
        self.draw_snake()
    
        # Draw snake
        #for (x, y) in self.snake:
        #    pygame.draw.rect(self.screen, COLORS["snake"], (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    
        # Draw food
        fx, fy = self.food
        pygame.draw.rect(self.screen, COLORS["food"], (fx * CELL_SIZE, fy * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        pygame.display.flip()

    def poll_input(self):
        """
        Note that direction can be controlled with arrow.
        Snake shall provide other method to control the snake
        when no UI is present.
        """
        if self.render is False:
            raise ValueError();
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            # Direction control
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP and self.direction != (0, 1):
                    self.direction = (0, -1)
                elif event.key == pygame.K_DOWN and self.direction != (0, -1):
                    self.direction = (0, 1)
                elif event.key == pygame.K_LEFT and self.direction != (1, 0):
                    self.direction = (-1, 0)
                elif event.key == pygame.K_RIGHT and self.direction != (-1, 0):
                    self.direction = (1, 0)

    def move(self):
        # Update snake
        head_x, head_y = self.snake[0]
        new_head = (head_x + self.direction[0], head_y + self.direction[1])

        # Check bounds
        if not (0 <= new_head[0] < GRID_WIDTH and 0 <= new_head[1] < GRID_HEIGHT):
            # LOOSE
            pygame.quit()
            sys.exit()

        # Check collision with snake
        if new_head in self.snake:
            # LOOSE
            pygame.quit()
            sys.exit()

        # Move snake
        self.snake.insert(0, new_head)
        if new_head == self.food:
            # Place new food
            while self.food in self.snake:
                self.food = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
        else:
            self.snake.pop()

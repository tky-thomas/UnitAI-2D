"""
File for generating the game map by spawning in both obstacles and game entities.
Responsible for maintaining a list of all entities,
and relaying this information to game objects.
Also, responsible for generating new enemy entities as they are killed.
"""
import math

import arcade.sprite
import numpy as np
import random
from entities import *

GRID_SIZE = 20

MAP_OBSTACLES = [[(5, 5, 9, 9), (25, 5, 7, 7), (5, 20, 5, 5), (20, 20, 5, 5)]]
MAP_ID = 0

ENEMY_COUNT = 10
SPAWN_RADIUS = 4

PLAYER = 1
ENEMY = 2
OBSTACLE = 3


class Environment:

    def __init__(self, window_width, window_height):
        self.window_width = window_width
        self.window_height = window_height

        # All environment information is represented as a 2D tuple
        self.grids_x = round(window_width / GRID_SIZE)
        self.grids_y = round(window_height / GRID_SIZE)
        self.grid = np.zeros((self.grids_y, self.grids_x))
        self.enemy_spawn_grid = None

        self.show_grid = False

        self.player = None
        self.obstacles = None
        self.enemies = None
        self.generate_world()

    def generate_world(self):
        # Spawns in the player at the center grid
        self.player = Player((round(self.grids_x / 2),
                              round(self.grids_y / 2)))

        # Generates the obstacles
        self.obstacles = arcade.SpriteList()
        for obstacle in MAP_OBSTACLES[MAP_ID]:
            obstacle_sprite = Obstacle((obstacle[0], obstacle[1]), obstacle[2], obstacle[3])
            self.obstacles.append(obstacle_sprite)

        # Gets an obstacle list and exclude enemy spawnpoints from these locations

        # Sets up a list to spawn enemies
        # Enemies will be spawned along the borders of the map
        self.enemy_spawn_grid = []
        for row in range(self.grids_y):
            for column in range(self.grids_x):
                if (column < SPAWN_RADIUS or column > (self.grids_x - SPAWN_RADIUS)) \
                        or (row < SPAWN_RADIUS or row > (self.grids_y - SPAWN_RADIUS)):
                    self.enemy_spawn_grid.append((column, row))

        # Generates an initial batch of enemy units
        self.enemies = arcade.SpriteList()
        for i in range(ENEMY_COUNT):
            # Pick a spawn location for the enemy
            enemy_sprite = Enemy(random.choice(self.enemy_spawn_grid))
            self.enemies.append(enemy_sprite)

    def draw(self):
        self.player.draw()
        for obstacle in self.obstacles:
            obstacle.draw()
        for enemy in self.enemies:
            enemy.draw()

        if self.show_grid:
            self.draw_grid()

    def update(self, delta_time):
        # Updates the player and enemies
        self.player.update()
        self.enemies.on_update(delta_time)

        # Updates the game map
        self.grid = self.get_map()

    def get_map(self):
        # Start with an empty grid
        grid = np.zeros((self.grids_y, self.grids_x))

        # Loops through all the grid positions, checking for obstacle collisions
        for obstacle in self.obstacles:
            grid_positions = get_grid_positions_box(obstacle)
            for pos in grid_positions:
                grid[pos[1]][pos[0]] = OBSTACLE

        # Loops through all the player and enemy entities
        x, y = get_grid_pos(self.player)
        grid[y][x] = PLAYER

        for enemy in self.enemies:
            x, y = get_grid_pos(enemy)
            grid[y][x] = ENEMY

        return grid

    def get_entity_list(self):
        pass

    def draw_grid(self):
        # Draws a gray-bordered grid to help visualize the map
        for i in range(0, self.window_height, GRID_SIZE):
            arcade.draw_line(0, i, self.window_width, i, color=arcade.color.GRAY, line_width=GRID_SIZE / 10)
        for i in range(0, self.window_width, GRID_SIZE):
            arcade.draw_line(i, 0, i, self.window_height, color=arcade.color.GRAY, line_width=GRID_SIZE / 10)

        # Draws a number of top of every entity on the grid
        for y, row in enumerate(self.grid):
            for x, column in enumerate(row):
                if column == 0:
                    continue
                arcade.draw_text(start_x=round((x * GRID_SIZE) + (GRID_SIZE / 2)),
                                 start_y=round((y * GRID_SIZE) + (GRID_SIZE / 2)),
                                 anchor_x="center", anchor_y="center",
                                 text=str(int(column)),
                                 color=arcade.color.GRAY)

    def toggle_grid(self):
        self.show_grid = not self.show_grid


def get_grid_pos(sprite: arcade.Sprite):
    return math.floor(sprite.center_x / GRID_SIZE), math.floor(sprite.center_y / GRID_SIZE)


def get_grid_positions_box(sprite):
    sprite_bottom_left_pos = (round((sprite.center_x - (sprite.width / 2)) / GRID_SIZE),
                              round((sprite.center_y - (sprite.height / 2)) / GRID_SIZE))
    sprite_top_right_pos = (round((sprite.center_x + (sprite.width / 2)) / GRID_SIZE),
                            round((sprite.center_y + (sprite.height / 2)) / GRID_SIZE))

    grid_positions = list()

    for row in range(sprite_bottom_left_pos[1], sprite_top_right_pos[1]):
        for column in range(sprite_bottom_left_pos[0], sprite_top_right_pos[0]):
            grid_positions.append((column, row))

    return grid_positions

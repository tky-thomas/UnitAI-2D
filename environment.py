"""
File for generating the game map by spawning in both obstacles and game entities.
Responsible for maintaining a list of all entities,
and relaying this information to game objects.
Also, responsible for generating new enemy entities as they are killed.
"""
import arcade.sprite
import numpy as np
from entities import Player, Obstacle

GRID_SIZE = 20

MAP_OBSTACLES = [[(5, 5, 9, 9), (25, 5, 7, 7), (5, 20, 5, 5), (20, 20, 5, 5)]]
MAP_ID = 0


class Environment:
    def __init__(self, window_width, window_height):
        self.window_width = window_width
        self.window_height = window_height

        # All environment information is represented as a 2D tuple
        self.grids_x = round(window_width / GRID_SIZE)
        self.grids_y = round(window_height / GRID_SIZE)
        self.grid = np.zeros((self.grids_y, self.grids_x))

        self.player = None
        self.obstacles = None
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

    def draw(self):
        self.player.draw()
        for obstacle in self.obstacles:
            obstacle.draw()

    def get_map(self):
        pass

    def get_entity_list(self):
        pass

    def get_gridsize(self):
        return round(self.window_width / len(self.grid[0]))

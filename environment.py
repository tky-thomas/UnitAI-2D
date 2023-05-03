"""
File for generating the game map by spawning in both obstacles and game entities.
Responsible for maintaining a list of all entities,
and relaying this information to game objects.
Also, responsible for generating new enemy entities as they are killed.
"""
import numpy as np
from entities import Player

GRID_SIZE = 20


class Environment:
    def __init__(self, window_width, window_height):
        self.window_width = window_width
        self.window_height = window_height

        # All environment information is represented as a 2D tuple
        self.grids_x = round(window_width / GRID_SIZE)
        self.grids_y = round(window_height / GRID_SIZE)
        self.grid = np.zeros((self.grids_y, self.grids_x))

        self.player = None
        self.generate_world()

    def generate_world(self):
        # Spawns in the player at the center grid
        self.player = Player((round(self.grids_x / 2),
                              round(self.grids_y / 2)))

    def draw(self):
        self.player.draw()

    def get_map(self):
        pass

    def get_entity_list(self):
        pass

    def get_gridsize(self):
        return round(self.window_width / len(self.grid[0]))

"""
File for generating the game map by spawning in both obstacles and game entities.
Responsible for maintaining a list of all entities,
and relaying this information to game objects.
Also responsible for generating new enemy entities as they are killed.
"""
import numpy as np

GRID_SIZE = 20

class Environment:
    def __init__(self, window_width, window_height):
        self.window_width = window_width
        self.window_height = window_height

        # All environment information is represented as a 2D tuple
        grids_x = round(window_width / GRID_SIZE)
        grids_y = round(window_height / GRID_SIZE)
        self.grid = np.zeros((grids_y, grids_x))

        print(len(self.grid[0]))

    def generate_world(self):
        pass

    def spawn_entity(self, grid_pos):
        pass

    def get_map(self):
        pass

    def get_entity_list(self):
        pass

    def get_gridsize(self):
        return round(self.window_width / len(self.grid[0]))
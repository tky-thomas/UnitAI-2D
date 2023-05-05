import arcade
import astar_pathfind as astar
import numpy as np

PLAYER = 1
ENEMY = 2
OBSTACLE = 3

class Player(arcade.Sprite):

    def __init__(self, spawn_pos_grid, grid_width=20, update_freq=1):
        super().__init__()
        self.pos = spawn_pos_grid
        self.update_freq = update_freq

        # Position the player
        self.height = grid_width
        self.width = grid_width
        self.center_x = (spawn_pos_grid[0] * grid_width) + round(grid_width / 2)
        self.center_y = (spawn_pos_grid[1] * grid_width) + round(grid_width / 2)

        self.grid_width = grid_width

    def draw(self, **kwargs):
        # Draws a red square
        arcade.draw_rectangle_filled(self.center_x, self.center_y, self.width, self.height,
                                     arcade.color.RED)


class Obstacle(arcade.Sprite):
    def __init__(self, spawn_pos_grid, grids_width, grids_height, grid_width=20):
        super().__init__()
        self.pos = spawn_pos_grid

        # Position the player
        self.width = grids_width * grid_width
        self.height = grids_height * grid_width
        self.center_x = (spawn_pos_grid[0] * grid_width) + (self.width / 2)
        self.center_y = (spawn_pos_grid[1] * grid_width) + (self.height / 2)

        self.grid_width = grid_width

    def draw(self, **kwargs):
        # Draws a yellow square
        arcade.draw_rectangle_filled(self.center_x, self.center_y, self.width, self.height,
                                     arcade.color.YELLOW)


class Enemy(arcade.Sprite):

    def __init__(self, spawn_pos_grid, attack_range=2, grid_width=20, update_freq=1):
        super().__init__()
        self.pos = spawn_pos_grid
        self.update_freq = update_freq
        self.update_timer = 0
        self.draw_path = False
        self.path = None

        # Position the enemy
        self.height = grid_width
        self.width = grid_width
        self.center_x = (spawn_pos_grid[0] * grid_width) + round(grid_width / 2)
        self.center_y = (spawn_pos_grid[1] * grid_width) + round(grid_width / 2)
        self.grid_width = grid_width

        self.grid = None

        # Enemy attributes
        self.range = attack_range

    def draw(self, **kwargs):
        arcade.draw_rectangle_filled(self.center_x, self.center_y, self.width, self.height,
                                     arcade.color.BLUE)
        # If indicated, draw the path to the player
        if self.draw_path and self.path is not None:
            for i in range(len(self.path)):
                # Do not draw for the last node
                if i + 1 == len(self.path):
                    continue
                current_step = self.path[i]
                next_step = self.path[i + 1]
                arcade.draw_line(((current_step[0] * self.grid_width) + self.grid_width / 2),
                                 ((current_step[1] * self.grid_width) + self.grid_width / 2),
                                 ((next_step[0] * self.grid_width) + self.grid_width / 2),
                                 ((next_step[1] * self.grid_width) + self.grid_width / 2),
                                 color=arcade.color.AERO_BLUE,
                                 line_width=2)
                arcade.draw_circle_filled(((current_step[0] * self.grid_width) + self.grid_width / 2),
                                          ((current_step[1] * self.grid_width) + self.grid_width / 2),
                                          radius=5,
                                          color=arcade.color.AERO_BLUE)

    def on_update(self, delta_time: float = 1 / 60):
        # Update timer
        self.update_timer += delta_time
        if self.update_timer < self.update_freq:
            return
        self.update_timer = 0

        # Pathfind to player
        target_pos = np.where(self.grid == PLAYER)
        target_pos = (target_pos[1], target_pos[0])
        self.path = astar.pathfind(self.grid, self.pos, target_pos, obstacles=(OBSTACLE,))

    def update_grid(self, grid: np.array):
        self.grid = grid

    def toggle_path_draw(self):
        self.draw_path = not self.draw_path




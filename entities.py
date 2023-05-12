import math

import arcade
import astar_pathfind as astar
import numpy as np

PLAYER = 1
ENEMY = 2
OBSTACLE = 3
UNIT_MARKER = 5

UP = 1
RIGHT = 2
DOWN = 3
LEFT = 4

PATHFIND_CYCLES = 3
ATTACK_RANGE = 6

class Player(arcade.Sprite):

    def __init__(self, spawn_pos_grid, grid_width=20, update_freq=1):
        super().__init__()
        self.pos = spawn_pos_grid
        self.update_freq = update_freq

        self.damage_received = 0

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

    def damage(self, damage):
        self.damage_received += damage


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

    def __init__(self, spawn_pos_grid, attack_range=ATTACK_RANGE, grid_width=20, update_freq=1, player=None):
        super().__init__()
        self.pos = spawn_pos_grid
        self.update_freq = update_freq
        self.update_timer = 0
        self.draw_path = False

        self.path = list()
        self.pathfind_cycles = PATHFIND_CYCLES
        self.pathfind_cycle_threshold = PATHFIND_CYCLES

        self.player = player

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
        if self.draw_path:
            for i in range(len(self.path)):
                # Do not draw a connecting line for the last node
                if i + 1 >= len(self.path):
                    continue
                current_step = self.path[i]
                next_step = self.path[i + 1]
                arcade.draw_line(((current_step[0] * self.grid_width) + self.grid_width / 2),
                                 ((current_step[1] * self.grid_width) + self.grid_width / 2),
                                 ((next_step[0] * self.grid_width) + self.grid_width / 2),
                                 ((next_step[1] * self.grid_width) + self.grid_width / 2),
                                 color=arcade.color.AERO_BLUE,
                                 line_width=1)
                arcade.draw_circle_filled(((current_step[0] * self.grid_width) + self.grid_width / 2),
                                          ((current_step[1] * self.grid_width) + self.grid_width / 2),
                                          radius=3,
                                          color=arcade.color.AERO_BLUE)

    def update(self, delta_time: float = 1 / 60):
        # Update timer
        self.update_timer += delta_time
        if self.update_timer < self.update_freq:
            return
        self.update_timer = 0
        self.pathfind_cycles += 1

        # Pathfind to player, reloading the pathfinder if necessary
        # TODO: This is a placeholder. Eventually the reinforcement AI will decide when to pathfind to player
        if self.pathfind_cycles >= self.pathfind_cycle_threshold:
            self.pathfind_cycles = 0
            target_pos = self.get_player_pos()
            self.path = astar.pathfind(self.grid, self.get_self_pos(), target_pos, obstacles=(OBSTACLE,))

        # Move along path, deleting trails
        # If not moving, enemies can damage the player if in range
        if len(self.path) > 0:
            move_pos = self.path[0]
            self.center_x = ((move_pos[0] * self.grid_width) + self.grid_width / 2)
            self.center_y = ((move_pos[1] * self.grid_width) + self.grid_width / 2)
            self.path.remove(self.path[0])
        else:
            if self.player_in_range():
                # Deal damage to the player
                self.player.damage(1)

    def update_with_action(self, action):

        # Move to the specified location by the action
        self_pos = self.get_self_pos()
        if action == UP:
            target_pos = (self_pos[0], self_pos[1] + 1)
        elif action == DOWN:
            target_pos = (self_pos[0], self_pos[1] - 1)
        elif action == RIGHT:
            target_pos = (self_pos[0] + 1, self_pos[1])
        elif action == LEFT:
            target_pos = (self_pos[0] - 1, self_pos[1])
        else:
            target_pos = (-1, -1)  # Not movable, signifies the enemy should do nothing

        if self.movable(target_pos):
            self.center_x = ((target_pos[0] * self.grid_width) + self.grid_width / 2)
            self.center_y = ((target_pos[1] * self.grid_width) + self.grid_width / 2)
        else:
            # TODO: LINE OF SIGHT
            if self.player_in_range():
                # Deal damage to the player
                self.player.damage(1)

    def update_grid(self, grid: np.array):
        self.grid = grid

    def get_personal_grid(self):
        """
        Returns a grid, but annotated with the personal position of this unit.
        Very importantly, used to feed the input of the neural net.
        :return: The environment grid, but with the personal position added in.
        """
        pos = self.get_self_pos()
        ret = self.grid
        ret[pos[1]][pos[0]] = UNIT_MARKER
        return ret

    def toggle_path_draw(self):
        self.draw_path = not self.draw_path

    def player_in_range(self):
        if self.get_distance_from_player() <= self.range:
            return True
        return False

    def get_distance_from_player(self):
        target_pos = self.get_player_pos()
        self_pos = self.get_self_pos()
        return math.sqrt(pow(target_pos[0] - self_pos[0], 2) + pow(target_pos[1] - self_pos[1], 2))

    def get_self_pos(self):
        return math.floor(self.center_x / self.grid_width), math.floor(self.center_y / self.grid_width)

    def get_player_pos(self):
        target_pos = np.where(self.grid == PLAYER)
        target_pos = (target_pos[1], target_pos[0])
        return target_pos

    def movable(self, target_pos):
        # Checks whether a particular grid square is a legal move
        if target_pos[1] >= len(self.grid) or target_pos[1] < 0:
            return False
        if target_pos[0] >= len(self.grid[0]) or target_pos[0] < 0:
            return False

        grid_object = self.grid[target_pos[1]][target_pos[0]]
        if grid_object == OBSTACLE or grid_object == PLAYER:
            return False

        return True

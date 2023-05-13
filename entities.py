import math
import random

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

PLAYER_RANGE = 6
PLAYER_AOE = 0


class Player(arcade.Sprite):

    def __init__(self, spawn_pos_grid, grid_width=20, update_freq=1, range=PLAYER_RANGE, aoe_range=PLAYER_AOE):
        super().__init__()
        self.pos = spawn_pos_grid
        self.update_freq = update_freq

        self.damage_received = 0

        self.range = range
        self.aoe_range = aoe_range
        self.prev_target = None

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

    def update_player(self, enemies):
        viable_targets = list()
        self.pos = xy_to_pos(self.center_x, self.center_y, self.grid_width)
        for enemy in enemies:
            # Check if in range
            enemy_pos = xy_to_pos(enemy.center_x, enemy.center_y, enemy.grid_width)
            if get_distance(self.pos, enemy_pos) <= self.range:
                viable_targets.append(enemy)

        # Select a target to destroy. This will be the one closest to the previous target.
        if len(viable_targets) > 0:
            targets = list()
            if self.prev_target is None:
                initial_target = random.choice(viable_targets)
            else:
                viable_targets.sort(
                    key=lambda s: get_distance(self.prev_target, xy_to_pos(s.center_x, s.center_y, s.grid_width)))
                initial_target = viable_targets[0]

            # AOE damage
            for enemy in enemies:
                # Check if in range
                enemy_pos = xy_to_pos(enemy.center_x, enemy.center_y, enemy.grid_width)
                if get_distance(xy_to_pos(initial_target.center_x, initial_target.center_y, initial_target.grid_width),
                                enemy_pos) <= self.aoe_range:
                    targets.append(enemy)

            self.prev_target = xy_to_pos(initial_target.center_x, initial_target.center_y, initial_target.grid_width)

            for target in targets:
                target.damage()


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
        self.damage_dealt = 0

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
            self.path = astar.pathfind(self.grid,
                                       xy_to_pos(self.center_x, self.center_y, self.grid_width),
                                       target_pos,
                                       obstacles=(OBSTACLE,))

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
                self.damage_dealt += 1

    def update_with_action(self, action):

        # Move to the specified location by the action
        self_pos = xy_to_pos(self.center_x, self.center_y, self.grid_width)
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
                # Deal damage to the player and add score
                self.damage_dealt += 1
                self.player.damage(1)

    def update_grid(self, grid: np.array):
        self.grid = grid

    def get_personal_grid(self):
        """
        Returns a grid, but annotated with the personal position of this unit.
        Very importantly, used to feed the input of the neural net.
        :return: The environment grid, but with the personal position added in.
        """
        pos = xy_to_pos(self.center_x, self.center_y, self.grid_width)
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
        self_pos = xy_to_pos(self.center_x, self.center_y, self.grid_width)
        return get_distance(target_pos, self_pos)

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

    def damage(self):
        # For now everything dies in a one hit kill
        self.remove_from_sprite_lists()


def get_distance(pos1, pos2):
    return math.sqrt(pow(pos1[0] - pos2[0], 2) + pow(pos1[1] - pos2[1], 2))


def xy_to_pos(center_x, center_y, grid_width):
    return math.floor(center_x / grid_width), math.floor(center_y / grid_width)


def los_exists(pos1, pos2, grid):

    # Find the leftmost position
    left_pos = pos1
    right_pos = pos2
    if pos1[0] > pos2[0]:
        left_pos = pos2
        right_pos = pos1

    x1 = left_pos[0]
    y1 = left_pos[1]
    x2 = right_pos[0]
    y2 = right_pos[1]

    m_new = 2 * (y2 - y1)
    slope_error_new = m_new - (x2 - x1)

    y = y1
    for x in range(x1, x2 + 1):

        print("(", x, ",", y, ")\n")

        # Add slope to increment angle formed
        slope_error_new = slope_error_new + m_new

        # Slope error reached limit, time to
        # increment y and update slope error.
        if slope_error_new >= 0:
            y = y + 1
            slope_error_new = slope_error_new - 2 * (x2 - x1)

    return True


if __name__ == "__main__":
    los_exists((8, 6), (1, 1), None)
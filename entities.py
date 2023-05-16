import math
import random

import arcade
import astar_pathfind as astar
import numpy as np

PLAYER = 10
ENEMY = 2
OBSTACLE = 6
UNIT_MARKER = 5

UP = 1
RIGHT = 2
DOWN = 3
LEFT = 4

C_SELF = 0
C_OBSTACLE = 1
C_ENEMY = 2
C_PLAYER = 3

PATHFIND_CYCLES = 3
ATTACK_RANGE = 6

PLAYER_RANGE = 6
PLAYER_AOE = 0


class Player(arcade.Sprite):

    def __init__(self, spawn_pos, grid_width=20, update_freq=1, range=PLAYER_RANGE, aoe_range=PLAYER_AOE):
        super().__init__()
        self.pos = spawn_pos
        self.update_freq = update_freq

        self.damage_received = 0

        self.range = range
        self.aoe_range = aoe_range
        self.prev_target = None

        # Position the player
        self.height = grid_width
        self.width = grid_width
        self.center_x = (spawn_pos[0] * grid_width) + round(grid_width / 2)
        self.center_y = (spawn_pos[1] * grid_width) + round(grid_width / 2)

        self.grid_width = grid_width

    def draw(self, **kwargs):
        # Draws a red square
        arcade.draw_rectangle_filled(self.center_x, self.center_y, self.width, self.height,
                                     arcade.color.RED)

    def damage(self, damage):
        self.damage_received += damage

    def update_player(self, enemies):
        viable_targets = list()
        self.pos = get_grid_pos(self)
        for enemy in enemies:
            # Check if in range
            enemy_pos = get_grid_pos(enemy)
            if get_distance(self.pos, enemy_pos) <= self.range:
                viable_targets.append(enemy)

        # Select a target to attack
        if len(viable_targets) > 0:
            targets = list()
            if self.prev_target is None:
                initial_target = random.choice(viable_targets)
            else:
                viable_targets.sort(
                    key=lambda s: get_distance(self.prev_target, get_grid_pos(s)))
                initial_target = viable_targets[0]

            # AOE damage
            for enemy in enemies:
                # Check if in range of AOE
                enemy_pos = get_grid_pos(enemy)
                if get_distance(get_grid_pos(initial_target), enemy_pos) <= self.aoe_range:
                    targets.append(enemy)

            # Previous target location is logged.
            # Player will prioritize attacking close to here in the next cycle.
            self.prev_target = get_grid_pos(initial_target)

            # Damages all targets in the AOE
            for target in targets:
                target.damage()


class Obstacle(arcade.Sprite):
    def __init__(self, spawn_pos, grids_width, grids_height, grid_width=20):
        super().__init__()
        self.pos = spawn_pos

        # Position the player
        self.width = grids_width * grid_width
        self.height = grids_height * grid_width
        self.center_x = (spawn_pos[0] * grid_width) + (self.width / 2)
        self.center_y = (spawn_pos[1] * grid_width) + (self.height / 2)

        self.grid_width = grid_width

    def draw(self, **kwargs):
        # Draws a yellow square
        arcade.draw_rectangle_filled(self.center_x, self.center_y, self.width, self.height,
                                     arcade.color.YELLOW)


class Enemy(arcade.Sprite):

    def __init__(self, spawn_pos, attack_range=ATTACK_RANGE, grid_width=20, update_freq=1, player=None):
        super().__init__()
        self.pos = spawn_pos
        self.update_freq = update_freq
        self.update_timer = 0
        self.draw_path = False
        self.health = 3

        self.path = list()
        self.pathfind_cycles = PATHFIND_CYCLES
        self.pathfind_cycle_threshold = PATHFIND_CYCLES

        self.player = player
        self.damage_dealt = 0

        # Position the enemy
        self.height = grid_width
        self.width = grid_width
        self.center_x = (spawn_pos[0] * grid_width) + round(grid_width / 2)
        self.center_y = (spawn_pos[1] * grid_width) + round(grid_width / 2)
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

    def update_astar(self, delta_time: float = 1 / 60):
        """
        Generic enemy behaviour will have it move toward the player and engage it once in range.
        :param delta_time:
        :return:
        """
        # Update timer
        self.update_timer += delta_time
        if self.update_timer < self.update_freq:
            return
        self.update_timer = 0
        self.pathfind_cycles += 1

        # Pathfind to player, reloading every few cycles
        if self.pathfind_cycles >= self.pathfind_cycle_threshold:
            self.pathfind_cycles = 0
            target_pos = get_grid_pos(self.player)
            self.path = astar.pathfind(self.grid,
                                       get_grid_pos(self),
                                       target_pos,
                                       obstacles=(OBSTACLE,))

        # Move along path, deleting trails
        # Attacks player as soon as in range
        if self.player_in_range():
            self.player.damage(1)
            self.damage_dealt += 1
        elif len(self.path) > 0:
            move_pos = self.path[0]
            self.center_x = ((move_pos[0] * self.grid_width) + self.grid_width / 2)
            self.center_y = ((move_pos[1] * self.grid_width) + self.grid_width / 2)
            self.path.remove(self.path[0])

    def update_qmove(self, action):
        """
        Makes a move according to a Q-Learning model's action.
        Just like A-Star pathfinding, can only damage the player when not moving.
        :param action:
        :return:
        """

        # Move to the specified location by the action
        self_pos = get_grid_pos(self)
        target_pos = (-1, -1)  # Not movable, signifies the enemy should do nothing
        if action == DOWN:
            target_pos = (self_pos[0], self_pos[1] - 1)
        if action == UP:
            target_pos = (self_pos[0], self_pos[1] + 1)
        if action == RIGHT:
            target_pos = (self_pos[0] + 1, self_pos[1])
        if action == LEFT:
            target_pos = (self_pos[0] - 1, self_pos[1])

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

    def toggle_path_draw(self):
        self.draw_path = not self.draw_path

    def player_in_range(self):
        """
        This check is separate from get_distance_from_player as the reward function relies on the actual distance
        between the enemy unit and the player.
        :return:
        """
        if self.get_distance_from_player() <= self.range:
            return True
        return False

    def get_distance_from_player(self):
        target_pos = get_grid_pos(self.player)
        self_pos = get_grid_pos(self)
        return get_distance(target_pos, self_pos)

    def movable(self, target_pos):
        """
        Checks whether a particular grid square is moveable.
        Obstacles, map edges and players are invalid moves
        :param target_pos:
        :return:
        """
        if target_pos[1] >= len(self.grid[0]) or target_pos[1] < 0:
            return False
        if target_pos[0] >= len(self.grid[0][0]) or target_pos[0] < 0:
            return False

        if self.grid[C_OBSTACLE][target_pos[1]][target_pos[0]] == 1 \
                or self.grid[C_PLAYER][target_pos[1]][target_pos[0]] == 1:
            return False

        return True

    def damage(self):
        # 3 HP to start
        self.health -= 1
        if self.health <= 0:
            self.remove_from_sprite_lists()


def get_distance(pos1, pos2):
    """
    Get the distance between two different grid spaces on the map.
    :param pos1:
    :param pos2:
    :return:
    """
    return math.sqrt(pow(pos1[0] - pos2[0], 2) + pow(pos1[1] - pos2[1], 2))


def los_exists(pos1, pos2, grid):
    # TODO: Incomplete feature
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


def get_grid_pos(sprite: arcade.Sprite):
    """
    Gets the positional (x, y) representation of a Sprite on the grid,
    where (x, y) represents the column and row position of the sprite, respectively.
    :param sprite:
    :return:
    """
    return math.floor(sprite.center_x / 20), math.floor(sprite.center_y / 20)


def get_grid_pos_box(sprite: arcade.Sprite):
    """
    Same as get_grid_pos, but extended for sprites larger than a single grid tile.
    :param sprite:
    :return:
    """
    sprite_bottom_left_pos = (round((sprite.center_x - (sprite.width / 2)) / 20),
                              round((sprite.center_y - (sprite.height / 2)) / 20))
    sprite_top_right_pos = (round((sprite.center_x + (sprite.width / 2)) / 20),
                            round((sprite.center_y + (sprite.height / 2)) / 20))

    grid_positions = list()

    for row in range(sprite_bottom_left_pos[1], sprite_top_right_pos[1]):
        for column in range(sprite_bottom_left_pos[0], sprite_top_right_pos[0]):
            grid_positions.append((column, row))

    return grid_positions


if __name__ == "__main__":
    los_exists((8, 6), (1, 1), None)
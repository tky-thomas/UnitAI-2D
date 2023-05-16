"""
File for generating the game map by spawning in both obstacles and game entities.
Responsible for maintaining a list of all entities,
and relaying this information to game objects.
Also, responsible for generating new enemy entities as they are killed.
"""
import arcade.sprite

import entities
from entities import *

GRID_SIZE = 20

MAP_OBSTACLES = [[(5, 5, 9, 9), (25, 5, 7, 7), (5, 20, 5, 5), (20, 20, 5, 5)]]
MAP_ID = 0

ENEMY_COUNT = 20
SPAWN_RADIUS = 4

# Multi-channel 2D grid representation used for neural network
NUM_CHANNELS = 4
C_SELF = 0
C_OBSTACLE = 1
C_ENEMY = 2
C_PLAYER = 3
ENEMY_SIGHT_RANGE = 7

# Single-channel grid object representation (legacy, but still used)
PLAYER = 10
ENEMY = 2
OBSTACLE = 6
ENEMY_FOCUSED = 15


class Environment:
    """
    Game environment. Handles state updates, rendering and reward feedback.
    """

    def __init__(self, window_width, window_height,
                 update_freq=1,
                 graphics_enabled=False,
                 player_enabled=False):
        self.window_width = window_width
        self.window_height = window_height
        self.update_freq = update_freq
        self.graphics_enabled = graphics_enabled
        self.player_enabled = player_enabled

        # Environment is internally represented as a multi-channel grid
        self.grids_x = round(window_width / GRID_SIZE)
        self.grids_y = round(window_height / GRID_SIZE)
        self.grid = np.zeros((NUM_CHANNELS, self.grids_y, self.grids_x))
        self.enemy_spawn_grid = None

        self.show_grid = False

        self.player = None
        self.obstacles = None
        self.enemies = None

        self.damage_text = None
        self.previous_player_damage = None

        self.generate_world()

    def generate_world(self):
        """
        Generates the game world and all its elements.
        :return:
        """
        # Spawns in the player at the center grid
        self.player = Player((round(self.grids_x / 2),
                              round(self.grids_y / 2)),
                             update_freq=self.update_freq)

        # Generates the obstacles
        self.obstacles = arcade.SpriteList()
        for obstacle in MAP_OBSTACLES[MAP_ID]:
            obstacle_sprite = Obstacle((obstacle[0], obstacle[1]), obstacle[2], obstacle[3])
            self.obstacles.append(obstacle_sprite)

        # Sets up a list of possible enemy spawn locations
        self.enemy_spawn_grid = []
        for row in range(self.grids_y):
            for column in range(self.grids_x):
                if (column < SPAWN_RADIUS or column > (self.grids_x - SPAWN_RADIUS)) \
                        or (row < SPAWN_RADIUS or row > (self.grids_y - SPAWN_RADIUS)):
                    self.enemy_spawn_grid.append((column, row))

        self.enemies = arcade.SpriteList()
        self.spawn_enemies(ENEMY_COUNT)

        # Updates grid knowledge and gives it to the AI enemies
        self.grid = self.get_map()
        for enemy in self.enemies:
            enemy.update_grid(self.grid)

        # Sets up a player damage text if graphics is enabled
        if self.graphics_enabled:
            self.damage_text = arcade.Text("",
                                           start_x=self.window_width / 15,
                                           start_y=self.window_height - (self.window_height / 10),
                                           color=arcade.color.AERO_BLUE,
                                           font_size=15)

        # Player damage tracker for reward calculation
        self.previous_player_damage = self.player.damage_received

    def draw(self):
        """
        Renders the game world.
        :return:
        """
        self.player.draw()
        for obstacle in self.obstacles:
            obstacle.draw()
        for enemy in self.enemies:
            enemy.draw()

        # Draw damage dealt to the player
        self.damage_text.text = "Damage: " + str(self.player.damage_received)
        self.damage_text.draw()

        if self.show_grid:
            self.draw_grid()

    def update(self, delta_time, action_list):
        # Updates the player and enemies
        self.player.update()

        # Updates enemy grid knowledge
        self.grid = self.get_map()
        for enemy in self.enemies:
            enemy.update_grid(self.grid)

        for i, enemy in enumerate(self.enemies):
            enemy.update_qmove(action_list[i])

        # Calculates the reward for this round
        rewards = self.calculate_reward()

        # The player now attacks, damaging a random enemy in range.
        # He will pick an enemy close to his previous target, with a small chance of randomly targeting instead.
        # This attack may have an AOE.
        if self.player_enabled:
            self.player.update_player(self.enemies)

        # Respawn dead enemies
        self.spawn_enemies(ENEMY_COUNT - len(self.enemies))

        # Updates the game map and player damage
        self.grid = self.get_map()
        self.previous_player_damage = self.player.damage_received

        return rewards

    def get_map(self):
        """
        Gets a multi-channel, binary map of the full game environment.
        :return: A numpy grid with dimensions (C, H, W) representing the game map. Each channel is a binary notation of
        whether its corresponding map feature is present in that grid square.
        """
        # Start with an empty grid
        grid = np.zeros((NUM_CHANNELS, self.grids_y, self.grids_x))

        # Position of obstacles
        for obstacle in self.obstacles:
            grid_positions = entities.get_grid_pos_box(obstacle)
            for pos in grid_positions:
                grid[C_OBSTACLE][pos[1]][pos[0]] = 1

        # Position of enemy entities
        for enemy in self.enemies:
            x, y = entities.get_grid_pos(enemy)
            grid[C_ENEMY][y][x] = 1

        # Position of player
        x, y = entities.get_grid_pos(self.player)
        grid[C_PLAYER][y][x] = 1

        return grid

    def get_state_maps(self):
        """
        Somewhat identical to get_map, but returns a list of state maps centered on each enemy, and
        restricted to their sight range.
        :return: A list of multi-channel binary state maps for all the enemy units, each centered on an enemy unit
        and extending to their sight range.
        Unlike get_map these sight maps can also include portions outside the game world,
        which are treated as obstacles.
        """
        world_map = self.get_map()
        state_maps = list()

        # Creates a map region according to the enemy sight range
        for enemy in self.enemies:
            # Generates a state map focused on the enemy unit
            state_map = np.zeros((NUM_CHANNELS, (ENEMY_SIGHT_RANGE * 2) + 1, (ENEMY_SIGHT_RANGE * 2) + 1))
            x, y = entities.get_grid_pos(enemy)
            start_x = x - ENEMY_SIGHT_RANGE
            start_y = y - ENEMY_SIGHT_RANGE

            # Centers the self-representation on the middle of this sight map
            state_map[C_SELF][ENEMY_SIGHT_RANGE][ENEMY_SIGHT_RANGE] = 1

            # Obstacles
            for i in range(len(state_map[C_OBSTACLE])):
                for j in range(len(state_map[C_OBSTACLE][i])):
                    pos = (start_x + j, start_y + i)
                    if pos[0] < 0 or pos[1] < 0 or pos[0] >= self.grids_x or pos[1] >= self.grids_y:
                        state_map[C_OBSTACLE][i][j] = 1
                    else:
                        if world_map[C_OBSTACLE][pos[1]][pos[0]] == 1:
                            state_map[C_OBSTACLE][i][j] = 1

            # Enemies
            for i in range(len(state_map[C_ENEMY])):
                for j in range(len(state_map[C_ENEMY][i])):
                    pos = (start_x + j, start_y + i)
                    if pos[0] < 0 or pos[1] < 0 or pos[0] >= self.grids_x or pos[1] >= self.grids_y:
                        continue
                    else:
                        if world_map[C_ENEMY][pos[1]][pos[0]] == 1:
                            state_map[C_ENEMY][i][j] = 1

            # Player
            player_on_map = False
            for i in range(len(state_map[C_PLAYER])):
                for j in range(len(state_map[C_PLAYER][i])):
                    pos = (start_x + j, start_y + i)
                    if pos[0] < 0 or pos[1] < 0 or pos[0] >= self.grids_x or pos[1] >= self.grids_y:
                        continue
                    else:
                        if world_map[C_PLAYER][pos[1]][pos[0]] == 1:
                            player_on_map = True
                            state_map[C_PLAYER][i][j] = 1

            # Puts the player on the map border if not in the map already
            if not player_on_map:
                # Find player pos
                player_pos = entities.get_grid_pos(self.player)
                x_diff = player_pos[0] - x
                y_diff = player_pos[1] - y
                if x_diff > enemy.range:
                    pmark_x = (enemy.range * 2)
                else:
                    pmark_x = 0
                if y_diff > enemy.range:
                    pmark_y = (enemy.range * 2)
                else:
                    pmark_y = 0
                state_map[C_PLAYER][pmark_y][pmark_x] = PLAYER
            state_maps.append(state_map)

        return state_maps

    def draw_grid(self):
        # TODO: Fix later. BUT DO NOT USE THE GRID MODE!

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

    def toggle_visual(self, key):
        if key == arcade.key.G:
            self.show_grid = not self.show_grid
        elif key == arcade.key.P:
            for enemy in self.enemies:
                enemy.toggle_path_draw()

    def calculate_reward(self):
        # Calculates the reward for every single enemy agent
        rewards = list()

        for enemy in self.enemies:
            # Reward damage on player
            reward = enemy.damage_dealt * 10
            enemy.damage_dealt = 0

            # Reward being close to player
            reward += 10 - enemy.get_distance_from_player()
            rewards.append(reward)

        return rewards

    def spawn_enemies(self, enemy_count):
        for i in range(enemy_count):
            enemy_sprite = Enemy(random.choice(self.enemy_spawn_grid), update_freq=self.update_freq, player=self.player)
            self.enemies.append(enemy_sprite)




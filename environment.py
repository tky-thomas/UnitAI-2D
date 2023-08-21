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
C_SELF = 0
C_OBSTACLE = 1
C_ENEMY = 2
C_PLAYER = 3
ENEMY_SIGHT_RANGE = 15

# Single-channel grid object representation (legacy, but still used)
PLAYER = 10
ENEMY = 2
OBSTACLE = 6
ENEMY_FOCUSED = 15


class Environment:

    def __init__(self, window_width, window_height, update_freq=1, graphics_enabled=False,
                 player_enabled=False, player_aoe=0):
        self.window_width = window_width
        self.window_height = window_height
        self.update_freq = update_freq
        self.graphics_enabled = graphics_enabled
        self.player_enabled = player_enabled
        self.player_aoe = player_aoe

        # All environment information is represented as a 2D tuple
        self.grids_x = round(window_width / GRID_SIZE)
        self.grids_y = round(window_height / GRID_SIZE)
        self.grid = np.zeros((self.grids_y, self.grids_x))
        self.enemy_spawn_grid = None

        self.show_grid = False

        self.player = None
        self.obstacles = None
        self.enemies = None

        self.damage_text = None
        self.previous_player_damage = None

        self.generate_world()

    def generate_world(self):
        # Spawns in the player at the center grid
        self.player = Player((round(self.grids_x / 2),
                              round(self.grids_y / 2)),
                             update_freq=self.update_freq,
                             aoe_range=self.player_aoe)

        # Generates the obstacles
        self.obstacles = arcade.SpriteList()
        for obstacle in MAP_OBSTACLES[MAP_ID]:
            obstacle_sprite = Obstacle((obstacle[0], obstacle[1]), obstacle[2], obstacle[3])
            self.obstacles.append(obstacle_sprite)

        # TODO: Gets an obstacle list and exclude enemy spawnpoints from these locations

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
        self.spawn_enemies(ENEMY_COUNT)

        # Updates grid knowledge
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
        self.player.draw()
        for obstacle in self.obstacles:
            obstacle.draw()
        for enemy in self.enemies:
            enemy.draw()

        # Draw player damage received
        self.damage_text.text = "Damage: " + str(self.player.damage_received)
        self.damage_text.draw()

        if self.show_grid:
            self.draw_grid()

    def update(self, delta_time, action_list):
        # Updates enemy grid knowledge
        self.grid = self.get_map()
        for enemy in self.enemies:
            enemy.update_grid(self.grid)

        for i, enemy in enumerate(self.enemies):
            # enemy.update(delta_time)
            enemy.update_with_action(action_list[i])

        # Calculates the reward for this round
        rewards = self.calculate_reward()

        # The player now attacks, removing a random enemy in range.
        # He will pick an enemy close to his previous target, with a random chance of switching targets.
        # This attack may have an AOE.
        # The scatter density of the AI around the point of impact is also recorded.
        # Scatter density will be None if nothing is engaged.
        scatter_density = None
        if self.player_enabled:
            scatter_density = self.player.update_player(self.enemies)

        # Respawn dead enemies
        self.spawn_enemies(ENEMY_COUNT - len(self.enemies))

        # Updates the game map and player damage
        self.grid = self.get_map()
        self.previous_player_damage = self.player.damage_received

        return rewards, scatter_density

    def get_map(self):
        # CHANNEL ORDER: Self, Obstacles/Map Bounds, Other Enemies, Player (or his minimapped position)

        # Start with an empty grid
        grid = np.zeros((self.grids_y, self.grids_x))

        # Loops through all the grid positions, checking for obstacle collisions
        for obstacle in self.obstacles:
            grid_positions = get_grid_positions_box(obstacle)
            for pos in grid_positions:
                grid[pos[1]][pos[0]] = OBSTACLE

        # Position of other enemies is given to the map output
        for enemy in self.enemies:
            x, y = get_grid_pos(enemy)
            grid[y][x] = ENEMY

        # Position of player
        x, y = get_grid_pos(self.player)
        grid[y][x] = PLAYER

        return grid

    def get_state_maps(self):
        world_map = self.get_map()
        state_maps = list()

        # Creates a 15x15 map region representing each unit's sight range
        for enemy in self.enemies:
            state_map = np.zeros((4, ENEMY_SIGHT_RANGE, ENEMY_SIGHT_RANGE))
            x, y = get_grid_pos(enemy)
            start_x = x - math.floor(ENEMY_SIGHT_RANGE / 2)  # 7 by default
            start_y = y - math.floor(ENEMY_SIGHT_RANGE / 2)

            # Generates a state map focused on the enemy unit
            # Centers the self-representation on the middle of this sight map
            state_map[C_SELF][math.floor(ENEMY_SIGHT_RANGE / 2)][math.floor(ENEMY_SIGHT_RANGE / 2)] = 1

            # Obstacles
            for i in range(len(state_map[C_OBSTACLE])):
                for j in range(len(state_map[C_OBSTACLE][i])):
                    pos = (start_x + j, start_y + i)
                    if pos[0] < 0 or pos[1] < 0 or pos[0] >= self.grids_x or pos[1] >= self.grids_y:
                        state_map[C_OBSTACLE][i][j] = 1
                    else:
                        if world_map[pos[1]][pos[0]] == OBSTACLE:
                            state_map[C_OBSTACLE][i][j] = 1

            # Enemies
            for i in range(len(state_map[C_ENEMY])):
                for j in range(len(state_map[C_ENEMY][i])):
                    pos = (start_x + j, start_y + i)
                    if pos[0] < 0 or pos[1] < 0 or pos[0] >= self.grids_x or pos[1] >= self.grids_y:
                        continue
                    else:
                        if world_map[pos[1]][pos[0]] == ENEMY:
                            state_map[C_ENEMY][i][j] = 1

            # Player
            player_on_map = False
            for i in range(len(state_map[C_PLAYER])):
                for j in range(len(state_map[C_PLAYER][i])):
                    pos = (start_x + j, start_y + i)
                    if pos[0] < 0 or pos[1] < 0 or pos[0] >= self.grids_x or pos[1] >= self.grids_y:
                        continue
                    else:
                        if world_map[pos[1]][pos[0]] == PLAYER:
                            player_on_map = True
                            state_map[C_PLAYER][i][j] = 1

            # Puts the player on the map border if not in the map already
            if not player_on_map:
                # Find player pos
                player_pos = entities.xy_to_pos(self.player.center_x, self.player.center_y, self.player.grid_width)
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

    def get_entity_list(self):
        pass

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
            # Distance reward is still there but less prominent
            reward += 5 - (enemy.get_distance_from_player() / 2)
            rewards.append(reward)

        return rewards

    def spawn_enemies(self, enemy_count):
        for i in range(enemy_count):
            # Pick a spawn location for the enemy
            enemy_sprite = Enemy(random.choice(self.enemy_spawn_grid), update_freq=self.update_freq, player=self.player)
            self.enemies.append(enemy_sprite)


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

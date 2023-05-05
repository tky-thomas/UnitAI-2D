import arcade


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
        # Represented as a blue square
        arcade.draw_rectangle_filled(self.center_x, self.center_y, self.width, self.height,
                                     arcade.color.BLUE)

    def on_update(self, delta_time: float = 1 / 60):
        # Check if should be attacking player
        # Check if at destination grid. If so, find new target
        # Check if should retarget
        pass

    def update_grid(self, grid):
        self.grid = grid




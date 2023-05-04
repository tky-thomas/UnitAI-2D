import arcade


class Grid:
    def __init__(self, window_width, window_height, grid_width):
        self.window_width = window_width
        self.window_height = window_height
        self.grid_width = grid_width
        self.toggle = False

    def draw(self):
        if self.toggle is False:
            return

        # Draws a gray-bordered grid to help visualize the map
        for i in range(0, self.window_height, self.grid_width):
            arcade.draw_line(0, i, self.window_width, i, color=arcade.color.GRAY, line_width=self.grid_width / 10)
        for i in range(0, self.window_width, self.grid_width):
            arcade.draw_line(i, 0, i, self.window_height, color=arcade.color.GRAY, line_width=self.grid_width / 10)

    def toggle_grid(self):
        self.toggle = not self.toggle


class Player(arcade.Sprite):

    def __init__(self, spawn_pos_grid, grid_width=20):
        super().__init__()
        self.pos = spawn_pos_grid

        # Position the player
        self.height = grid_width
        self.width = grid_width
        self.center_x = (spawn_pos_grid[0] * grid_width) + round(grid_width / 2)
        self.center_y = (spawn_pos_grid[1] * grid_width) + round(grid_width / 2)

        self.grid_width = grid_width

    def draw(self, filter=None, pixelated=None, blend_function=None):
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


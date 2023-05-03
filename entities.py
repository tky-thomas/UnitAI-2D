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

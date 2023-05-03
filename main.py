"""
Starting Template

Once you have learned how to use classes, you can begin your program with this
template.

If Python and Arcade are installed, this example can be run from the command line with:
python -m arcade.examples.starting_template
"""
import arcade
from environment import Environment
from entities import Grid

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
WINDOW_TITLE = "UnitAI2D"


class UnitAI2D(arcade.Window):

    def __init__(self, width, height, title):
        super().__init__(width, height, title)
        arcade.set_background_color(arcade.color.AMAZON)

        self.environment = None

        # The game grid
        self.gridEntity = None

    def setup(self):
        """ Set up the game variables. Call to re-start the game. """
        self.environment = Environment(self.width, self.height)
        self.gridEntity = Grid(self.width, self.height, self.environment.get_gridsize())

    def on_draw(self):
        """
        Render the screen.
        """
        self.clear()

        # Call draw() on all your sprite lists below
        self.gridEntity.draw()

        # The Environment holds all entities, so we draw here
        self.environment.draw()

    def on_update(self, delta_time):
        """
        All the logic to move, and the game logic goes here.
        Normally, you'll call update() on the sprite lists that
        need it.
        """
        pass

    def on_key_press(self, key, key_modifiers):

        # Toggles the grid
        if key is arcade.key.G:
            self.gridEntity.toggle_grid()

    def on_key_release(self, key, key_modifiers):
        """
        Called whenever the user lets off a previously pressed key.
        """
        pass

    def on_mouse_motion(self, x, y, delta_x, delta_y):
        """
        Called whenever the mouse moves.
        """
        pass

    def on_mouse_press(self, x, y, button, key_modifiers):
        """
        Called when the user presses a mouse button.
        """
        pass

    def on_mouse_release(self, x, y, button, key_modifiers):
        """
        Called when a user releases a mouse button.
        """
        pass


def main():
    """ Main function """
    game = UnitAI2D(WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_TITLE)
    game.setup()
    arcade.run()


if __name__ == "__main__":
    main()

"""
Starting Template

Once you have learned how to use classes, you can begin your program with this
template.

If Python and Arcade are installed, this example can be run from the command line with:
python -m arcade.examples.starting_template
"""
import arcade
from environment import Environment

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
WINDOW_TITLE = "UnitAI2D"


class UnitAI2D(arcade.Window):

    def __init__(self, width, height, title):
        super().__init__(width, height, title)
        arcade.set_background_color(arcade.color.AMAZON)

        self.environment = None

    def setup(self):
        """ Set up the game variables. Call to re-start the game. """
        self.environment = Environment(self.width, self.height)

    def on_draw(self):
        """
        Render the screen.
        """
        self.clear()
        self.environment.draw()

    def on_update(self, delta_time):
        """
        All the logic to move, and the game logic goes here.
        Normally, you'll call update() on the sprite lists that
        need it.
        """
        self.environment.update(delta_time)

    def on_key_press(self, key, key_modifiers):

        # Toggles the grid
        self.environment.toggle_visual(key)


def main():
    """ Main function """
    game = UnitAI2D(WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_TITLE)
    game.setup()
    arcade.run()


if __name__ == "__main__":
    main()

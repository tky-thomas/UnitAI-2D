"""
Starting Template

Once you have learned how to use classes, you can begin your program with this
template.

If Python and Arcade are installed, this example can be run from the command line with:
python -m arcade.examples.starting_template
"""
import arcade
from environment import Environment
import random

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from deep_q_learning import DeepQNetwork_FullMap
from replay_memory import DeepQReplay

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
WINDOW_TITLE = "UnitAI2D"

UPDATE_FREQUENCY = 1

# Machine learning hyperparameters
LR = 0.01
EPSILON = 0.3
EPSILON_DECAY_RATE = 0.9
MEMORY_CAPACITY = 1000

# Random Seed
RANDOM_SEED = None


class UnitAI2D(arcade.Window):

    def __init__(self, width, height, title):
        super().__init__(width, height, title)
        arcade.set_background_color(arcade.color.AMAZON)

        self.environment = None
        self.update_timer = None
        self.update_freq = None

        # Deep Q Learning model
        self.model = None
        self.device = None
        self.optimizer = None
        self.memory = None

    def setup(self):
        """ Set up the game variables. Call to re-start the game. """
        self.update_timer = 0
        self.update_freq = UPDATE_FREQUENCY
        self.environment = Environment(self.width, self.height, self.update_freq)

        self.machine_learning_setup()

    def on_draw(self):
        """
        Render the screen.
        """
        self.clear()
        self.environment.draw()

    def on_update(self, delta_time):
        # Update timer
        self.update_timer += delta_time
        if self.update_timer < self.update_freq:
            return
        self.update_timer = 0

        # Model returns information about all units
        state_maps = self.environment.get_state_maps()

        # For each unit, the model selects an action
        action_list = list()
        for state_map in state_maps:
            # Convert each state map into a tensor
            state_map = torch.Tensor(state_map)
            state_map = state_map.unsqueeze(0)

            action_tensor = self.model(state_map)
            action_list.append(self.model.action_translate(action_tensor))

        # Action is fed back into the network
        reward = self.environment.update(delta_time, action_list)

        # Saves the state, action, reward and next state
        for state_map in state_maps:
            self.memory.push()

    def on_key_press(self, key, key_modifiers):

        # Toggles the grid
        self.environment.toggle_visual(key)

    def machine_learning_setup(self):
        # ====================
        # REPLAY MEMORY (DATASET)
        # ====================
        self.memory = DeepQReplay(capacity=MEMORY_CAPACITY)

        # ====================
        # MODEL
        # ====================
        self.model = DeepQNetwork_FullMap(random_action_chance=EPSILON, random_decay_rate=EPSILON_DECAY_RATE)
        print('\nModel:')
        print(self.model)

        use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        print('Device: {}'.format(self.device))

        if use_cuda:
            cudnn.enabled = True
            cudnn.benchmark = True
            print('CUDNN is enabled. CUDNN benchmark is enabled')
            self.model.cuda()

        params = [p.nelement() for p in self.model.parameters() if p.requires_grad]
        num_params = sum(params)

        print('num_params:', num_params)
        print(flush=True)

        # ====================
        # OPTIMIZER
        # ====================
        parameters = filter(lambda x: x.requires_grad, self.model.parameters())
        optimizer = optim.Adam(parameters, lr=LR)
        print(optimizer)


def main():
    random.seed(RANDOM_SEED)

    """ Main function """
    game = UnitAI2D(WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_TITLE)
    game.setup()
    arcade.run()


if __name__ == "__main__":
    main()

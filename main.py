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
import torch.nn as nn
from deep_q_learning import DeepQNetwork_FullMap
from replay_memory import DeepQReplay, StateTransition

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
WINDOW_TITLE = "UnitAI2D"

UPDATE_FREQUENCY = 0.1

# Machine learning hyperparameters
EPISODE_CYCLES = 100
LR = 0.01
EPSILON = 0.9
EPSILON_DECAY_RATE = 0.9
MEMORY_CAPACITY = 1000
BATCH_SIZE = 32
GAMMA = 0.99

# Random Seed
RANDOM_SEED = None


class UnitAI2D(arcade.Window):

    def __init__(self, width, height, title, cycles=300, training_batch_size=32, gamma=0.99):
        super().__init__(width, height, title)
        arcade.set_background_color(arcade.color.AMAZON)

        self.environment = None
        self.update_timer = None
        self.cycle_timer = None
        self.update_freq = None

        # Deep learning model setup
        self.policy_model = None
        self.target_model = None
        self.device = None
        self.optimizer = None
        self.memory = None
        self.training_batch_size = training_batch_size
        self.gamma = gamma

        self.episode_cycles = cycles
        self.machine_learning_setup()

    def setup(self):
        """ Set up the game variables. Call to re-start the game. """
        self.update_timer = 0
        self.cycle_timer = 0
        self.update_freq = UPDATE_FREQUENCY
        self.environment = Environment(self.width, self.height, self.update_freq)

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

        # Cycle counter. If the number of cycles exceeds a certain number,
        # then initiate network training and reset the game environment
        self.cycle_timer += 1
        if self.cycle_timer > self.episode_cycles:
            self.train_network()
            self.setup()

        # Model returns information about all units
        state_maps = self.environment.get_state_maps()

        # For each unit, the model selects an action
        action_list = list()
        for state_map in state_maps:
            # Convert each state map into a tensor
            state_map = torch.Tensor(state_map).unsqueeze(0)

            action_tensor = self.policy_model(state_map)
            action_list.append(self.policy_model.action_translate(action_tensor))

        # Action is fed back into the network
        reward = self.environment.update(delta_time, action_list)

        # Gets the next state information from the network
        next_state_maps = self.environment.get_state_maps()

        # Saves the state, action, reward and next state for training.
        # There are many agents, so only five random agent's states are saved
        for i in range(10):
            state_idx = random.randint(0, len(state_maps) - 1)
            state_t = torch.tensor(state_maps[state_idx], dtype=torch.float32, device=self.device)
            action_t = action_list[state_idx].reshape(1,)
            next_state_t = torch.tensor(next_state_maps[state_idx], dtype=torch.float32, device=self.device)
            reward_t = torch.tensor([reward], dtype=torch.float32, device=self.device)
            self.memory.push(state_t, action_t, reward_t, next_state_t)

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
        self.policy_model = DeepQNetwork_FullMap(random_action_chance=EPSILON, random_decay_rate=EPSILON_DECAY_RATE)
        self.target_model = DeepQNetwork_FullMap(random_action_chance=EPSILON, random_decay_rate=EPSILON_DECAY_RATE)
        self.target_model.load_state_dict(self.policy_model.state_dict())
        print('\nModel:')
        print(self.policy_model)

        use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        print('Device: {}'.format(self.device))

        if use_cuda:
            cudnn.enabled = True
            cudnn.benchmark = True
            print('CUDNN is enabled. CUDNN benchmark is enabled')
            self.policy_model.cuda()
            self.target_model.cuda()

        params = [p.nelement() for p in self.policy_model.parameters() if p.requires_grad]
        num_params = sum(params)

        print('num_params:', num_params)
        print(flush=True)

        # ====================
        # OPTIMIZER
        # ====================
        parameters = filter(lambda x: x.requires_grad, self.policy_model.parameters())
        self.optimizer = optim.Adam(parameters, lr=LR)
        print(self.optimizer)

        # ====================
        # MODEL SAVING
        # ====================

    def train_network(self):
        # TODO: CLEAN UP COMMENTS
        # Gets a sample from the memory
        if len(self.memory) < BATCH_SIZE:
            return
        training_batch = self.memory.sample(BATCH_SIZE)
        batch = StateTransition(*zip(*training_batch))

        # Technically there will always be a next state in this infinite simulation,
        # so finding a non-final mask is not needed
        state_batch = torch.stack(batch.state)
        action_batch = torch.stack(batch.action)
        reward_batch = torch.stack(batch.reward)
        next_state_batch = torch.stack(batch.next_state)

        print("Average Reward:", (torch.sum(reward_batch) / self.training_batch_size).item())

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_model(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.training_batch_size, device=self.device)
        with torch.no_grad():
            next_state_values = self.target_model(next_state_batch).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_model.parameters(), 100)
        self.optimizer.step()


def main():
    random.seed(RANDOM_SEED)

    """ Main function """
    game = UnitAI2D(WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_TITLE, EPISODE_CYCLES, BATCH_SIZE, GAMMA)
    game.setup()
    arcade.run()


if __name__ == "__main__":
    main()

import math
import os

import arcade
from environment import Environment
import random
import time
import sys
import pickle

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn
from deep_q_learning import DeepQNetwork_FullMap
from replay_memory import DeepQReplay, StateTransition

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
WINDOW_TITLE = 'UnitAI2D'
ENABLE_GRAPHICS = True

UPDATE_FREQUENCY = 0.05

ENABLE_PLAYER = True

# Machine learning hyperparameters
NUM_EPISODES = 200
EPISODE_CYCLES = 150

# LEGACY: Code of the Epsilon-greedy step selection preserved for reference.
# It did not work.
# EPSILON_START = 0.9
# EPSILON_END = 0.05
# EPSILON_DECAY_RATE = NUM_EPISODES / 4

MEMORY_CAPACITY = 2048
BATCH_SIZE = 1024
GAMMA = 0.99  # Coefficient of future action value
LR = 0.00001  # Learning rate of policy network.
TAU = 0.005  # Rate at which target network is updated - separate target and policy networks create smoother training

# Files
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

PROGRAM_MODE = 'train'  # OPTIONS: 'train', 'simulate'
LOAD_MODEL = True
SAVE_MODEL = True
SAVE_RESULT = True

POLICY_MODEL_LOAD_PATH = os.path.join(__location__, 'saved_models/unit_ai_2d_policy2_2.pt')
POLICY_MODEL_SAVE_PATH = os.path.join(__location__, 'saved_models/unit_ai_2d_policy2_3.pt')
TARGET_MODEL_LOAD_PATH = os.path.join(__location__, 'saved_models/unit_ai_2d_target2_2.pt')
TARGET_MODEL_SAVE_PATH = os.path.join(__location__, 'saved_models/unit_ai_2d_target2_3.pt')
RESULT_SAVE_PATH = os.path.join(__location__, 'results/120523_no_death_ranged2_3.pt')
SIMULATION_LOAD_PATH = os.path.join(__location__, 'saved_models/unit_ai_2d_policy2_3.pt')

# Random Seed
RANDOM_SEED = None


class UnitAI2D:
    def __init__(self, width, height,
                 program_mode='train',
                 graphics_enabled=False,
                 num_episodes=100, episode_cycles=200,
                 update_freq=0.05,
                 episode_sample_size=10,
                 training_batch_size=32,
                 gamma=0.99, lr=0.0001, tau=0.005,
                 load_model=False, save_model=False,
                 policy_model_load_path=None, policy_model_save_path=None,
                 target_model_load_path=None, target_model_save_path=None,
                 simulation_load_path=None,
                 enable_player=False):
        self.width = width
        self.height = height
        self.graphics_enabled = graphics_enabled
        self.environment = None
        self.enable_player = enable_player

        self.num_episodes = num_episodes
        self.episode_counter = 1
        self.update_freq = update_freq
        self.update_timer = None
        self.episode_cycles = episode_cycles
        self.episode_cycle_counter = None

        # Deep learning model
        self.program_mode = program_mode
        self.policy_model = None
        self.target_model = None
        self.device = None
        self.optimizer = None
        self.memory = None

        self.episode_sample_size = episode_sample_size
        self.training_batch_size = training_batch_size
        self.gamma = gamma
        self.lr = lr
        self.tau = tau

        # Model loading and saving
        self.load_model = load_model
        self.save_model = save_model
        self.policy_model_load_path = policy_model_load_path
        self.policy_model_save_path = policy_model_save_path
        self.target_model_load_path = target_model_load_path
        self.target_model_save_path = target_model_save_path
        self.simulation_load_path = simulation_load_path

        # Saving episode data
        self.network_losses = list()
        self.reward_history = list()
        self.damage_history = list()
        self.avg_scatter_density_history = list()
        self.episode_scatter_densities = list()

        self.machine_learning_setup()

    def setup(self):
        """
        Resets the environment at the beginning of each episode.
        :return:
        """
        self.update_timer = 0
        self.episode_cycle_counter = 0
        self.episode_scatter_densities = list()
        self.environment = \
            Environment(self.width, self.height, self.update_freq, self.graphics_enabled, self.enable_player)

    def on_update(self, delta_time):
        """
        Updates the environment on every cycle of an episode.
        Can perform updates according to a delta_time, where one cycle is performed according to update_freq.
        For low update_freq, the computation time for machine learning usually exceeds update_freq,
        causing actual update frequency to be lower.
        :param delta_time: Time elapsed since the last time this function was called.
        :return:
        """
        # Update timer
        self.update_timer += delta_time
        if self.update_timer < self.update_freq:
            return
        self.update_timer = 0

        # Episode cycle counter
        # Initiates one training step when the episode has concluded.
        self.episode_cycle_counter += 1
        if self.episode_cycle_counter > self.episode_cycles:

            if self.program_mode == 'train':
                network_loss, total_reward = self.train_network()
                self.network_losses.append(network_loss)
                self.reward_history.append(total_reward)
            self.damage_history.append(self.environment.previous_player_damage)
            self.avg_scatter_density_history.append(sum(self.episode_scatter_densities)
                                                    / len(self.episode_scatter_densities))

            self.stdout_write("Damage: %d  Avg. Scatter Density %.2f  "
                              % (self.environment.previous_player_damage,
                                 sum(self.episode_scatter_densities) / len(self.episode_scatter_densities)))
            self.stdout_write("\n")

            self.episode_counter += 1
            if self.episode_counter > self.num_episodes:
                return True
            self.setup()

        self.draw_progress_bar()

        # State-action loop.
        state_maps = self.environment.get_state_maps()
        action_list = list()
        for state_map in state_maps:
            # Pads the batch size dimension
            state_map = torch.Tensor(state_map).unsqueeze(0).to(self.device)

            # The model does not train the layers translating the raw output of the network into an action
            action_t = self.policy_model(state_map)
            action_list.append(self.policy_model.action_translate(action_t))

        rewards, scatter_density = self.environment.update(delta_time, action_list)
        if scatter_density is not None:
            self.episode_scatter_densities.append(scatter_density)

        next_state_maps = self.environment.get_state_maps()

        # Samples transitions from several random agents.
        if self.program_mode == 'train':
            for i in range(self.episode_sample_size):
                sample_index = random.randint(0, len(state_maps) - 1)

                state_t = torch.tensor(state_maps[sample_index], dtype=torch.float32, device=self.device)
                next_state_t = torch.tensor(next_state_maps[sample_index], dtype=torch.float32, device=self.device)

                # Pads the raw int with an extra dimension before tensor conversion
                action_t = torch.tensor([action_list[sample_index]], dtype=torch.int64, device=self.device)
                reward_t = torch.tensor([rewards[sample_index]], dtype=torch.float32, device=self.device)

                self.memory.push(state_t, action_t, reward_t, next_state_t)

        # Return False to continue training
        return False

    def machine_learning_setup(self):
        """
        Additional code to setup the machine learning models,
        separated from the init method for convenience.
        :return:
        """
        # ====================
        # REPLAY MEMORY (DATASET)
        # ====================
        self.memory = DeepQReplay(capacity=MEMORY_CAPACITY)

        # ====================
        # MODEL
        # ====================
        if self.save_model and self.program_mode == 'train':
            if not os.path.isdir(os.path.dirname(self.policy_model_save_path)):
                os.makedirs(os.path.dirname(self.policy_model_save_path))

        self.policy_model = DeepQNetwork_FullMap()
        self.target_model = DeepQNetwork_FullMap()

        # CUDA and training device settings
        use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        print('Device: {}'.format(self.device))

        if use_cuda:
            cudnn.enabled = True
            cudnn.benchmark = True
            print('CUDNN is enabled. CUDNN benchmark is enabled')
            self.policy_model.cuda()
            self.target_model.cuda()

        # Initializes the target model to be the same as the training model
        self.target_model.load_state_dict(self.policy_model.state_dict())

        # Model loading
        if self.load_model:
            self.policy_model.load_state_dict(torch.load(self.policy_model_load_path, map_location=self.device))
            self.target_model.load_state_dict(torch.load(self.target_model_load_path, map_location=self.device))
        if self.program_mode == 'simulate':
            self.policy_model.load_state_dict(torch.load(self.simulation_load_path, map_location=self.device))

        print('\nModel:')
        print(self.policy_model)

        # Parameter counting
        params = [p.nelement() for p in self.policy_model.parameters() if p.requires_grad]
        num_params = sum(params)
        self.stdout_write('num_params: %d\n' % num_params)

        # ====================
        # OPTIMIZER
        # ====================
        parameters = filter(lambda x: x.requires_grad, self.policy_model.parameters())
        self.optimizer = optim.Adam(parameters, lr=LR)
        print(self.optimizer)

    def train_network(self):
        """
        Training pipeline to perform a training step on the neural network.
        :return:
        """
        # Samples a training batch from memory, converting it to useful form
        if len(self.memory) < BATCH_SIZE:
            return
        training_batch = self.memory.sample(BATCH_SIZE)
        batch = StateTransition(*zip(*training_batch))

        state_batch = torch.stack(batch.state)
        action_batch = torch.stack(batch.action)
        reward_batch = torch.stack(batch.reward)
        next_state_batch = torch.stack(batch.next_state)

        # Compute the Q-function values for this state and the max to train the model
        state_action_values = self.policy_model(state_batch).gather(1, action_batch)
        with torch.no_grad():
            # Selecting the max Q trains the policy net to approximate the max
            next_state_values = self.target_model(next_state_batch).max(1)[0].unsqueeze(1)
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Huber Loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)
        self.stdout_write("Total Reward: %.2f  Loss: %.5f  "
                          % (torch.sum(reward_batch).item(),
                             loss.item()))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_model.parameters(), 100)
        self.optimizer.step()

        # Updates the target to better match policy according self.tau
        target_model_state_dict = self.target_model.state_dict()
        policy_model_state_dict = self.policy_model.state_dict()
        for key in policy_model_state_dict:
            target_model_state_dict[key] = (policy_model_state_dict[key] * self.tau) \
                                           + (target_model_state_dict[key] * (1 - self.tau))
        self.target_model.load_state_dict(target_model_state_dict)

        # Save the policy and target model
        if self.save_model:
            torch.save(self.policy_model.state_dict(), self.policy_model_save_path)
            torch.save(self.target_model.state_dict(), self.target_model_save_path)
            self.stdout_write("Model Saved  ")

        # Return the training loss and reward achieved
        return loss.item(), torch.sum(reward_batch).item()

    def draw_progress_bar(self):
        bars = math.floor(self.episode_cycle_counter * 20 / self.episode_cycles)
        self.stdout_write('\r', "Episode %d/%d: [%-20s] %d%%    %d | %d    "
                          % (self.episode_counter,
                             self.num_episodes,
                             '=' * bars,
                             round(self.episode_cycle_counter * 100 / self.episode_cycles),
                             self.episode_cycle_counter,
                             self.episode_cycles))

    @staticmethod
    def stdout_write(*args):
        for text in args:
            sys.stdout.write(text)
        sys.stdout.flush()
        time.sleep(0.001)


class UnitAI2D_Window(arcade.Window):
    """
    Simply wraps a Python Arcade window around the training loop,
    to allow for the game environment to be displayed.
    """
    def __init__(self, title, ai2d):
        super().__init__(ai2d.width, ai2d.height, title)
        arcade.set_background_color(arcade.color.AMAZON)

        self.ai2d = ai2d

    def setup(self):
        self.ai2d.setup()

    def on_draw(self):
        self.clear()
        self.ai2d.environment.draw()

    def on_update(self, delta_time):
        # Close window when simulation is complete
        if self.ai2d.on_update(delta_time):
            self.close()

    def on_key_press(self, key, key_modifiers):
        # Toggles the grid
        self.ai2d.environment.toggle_visual(key)


def main(graphics_mode=ENABLE_GRAPHICS):
    random.seed(RANDOM_SEED)

    ai2d = UnitAI2D(WINDOW_WIDTH, WINDOW_HEIGHT,
                    program_mode=PROGRAM_MODE,
                    num_episodes=NUM_EPISODES, episode_cycles=EPISODE_CYCLES,
                    update_freq=UPDATE_FREQUENCY,
                    training_batch_size=BATCH_SIZE, gamma=GAMMA, lr=LR, tau=TAU,
                    load_model=LOAD_MODEL, save_model=SAVE_MODEL,
                    policy_model_load_path=POLICY_MODEL_LOAD_PATH,
                    policy_model_save_path=POLICY_MODEL_SAVE_PATH,
                    target_model_load_path=TARGET_MODEL_LOAD_PATH,
                    target_model_save_path=TARGET_MODEL_SAVE_PATH,
                    simulation_load_path=SIMULATION_LOAD_PATH,
                    graphics_enabled=ENABLE_GRAPHICS,
                    enable_player=ENABLE_PLAYER)

    # If the display is enabled, arcade will run the game.
    # Otherwise, the update cycle will run without a window and draw
    if ENABLE_GRAPHICS:
        game = UnitAI2D_Window(WINDOW_TITLE, ai2d)
        game.setup()
        arcade.run()
    else:
        ai2d.setup()
        prev_time = time.time()
        while True:
            current_time = time.time()
            delta_time = current_time - prev_time
            if ai2d.on_update(delta_time):
                break
            prev_time = current_time

    # Save Episode Results
    if SAVE_RESULT:
        if not os.path.isdir(os.path.dirname(RESULT_SAVE_PATH)):
            os.makedirs(os.path.dirname(RESULT_SAVE_PATH))

    result = {'loss': ai2d.network_losses,
              'reward': ai2d.reward_history}
    with open(RESULT_SAVE_PATH, 'wb') as result_file:
        pickle.dump(result, result_file, protocol=pickle.HIGHEST_PROTOCOL)

    print("Training complete!")


if __name__ == "__main__":
    main()

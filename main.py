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
WINDOW_TITLE = "UnitAI2D"
GRAPHICS_MODE = "display"  # OPTIONS: display, no-display

UPDATE_FREQUENCY = 0.05

# Machine learning hyperparameters
NUM_EPISODES = 100
EPISODE_CYCLES = 300

EPSILON_START = 0.0
EPSILON_END = 0.0
EPSILON_DECAY_RATE = NUM_EPISODES / 2

MEMORY_CAPACITY = 2000
BATCH_SIZE = 512
GAMMA = 0.99  # Coefficient of future action value
LR = 0.001
TAU = 0.2  # Rate at which target network is updated

# Scenario Update
ENABLE_PLAYER = False

# Files
POLICY_MODEL_LOAD_PATH = "saved_models/unit_ai_2d_policy_2.pt"
POLICY_MODEL_SAVE_PATH = "saved_models/unit_ai_2d_policy_3.pt"
TARGET_MODEL_LOAD_PATH = "saved_models/unit_ai_2d_target_2.pt"
TARGET_MODEL_SAVE_PATH = "saved_models/unit_ai_2d_target_3.pt"
RESULT_SAVE_PATH = "results/120523_no_death_ranged_2.pt"
LOAD_MODEL = True
SAVE_MODEL = False
SAVE_RESULT = True

# Random Seed
RANDOM_SEED = None


class UnitAI2D:
    def __init__(self, width, height, training_batch_size=32, gamma=0.99, graphics_enabled=False,
                 num_episodes=100, episode_cycles=200,
                 load_model=False, save_model=False,
                 policy_model_load_path=None, policy_model_save_path=None,
                 target_model_load_path=None, target_model_save_path=None,
                 enable_player=False):
        self.environment = None
        self.update_timer = None
        self.cycle_timer = None
        self.episode_timer = 1
        self.update_freq = None
        self.width = width
        self.height = height
        self.graphics_enabled = graphics_enabled

        # Deep learning model setup
        self.policy_model = None
        self.target_model = None
        self.device = None
        self.optimizer = None
        self.memory = None

        # Model loading and saving
        self.load_model = load_model
        self.save_model = save_model
        self.policy_model_load_path = policy_model_load_path
        self.policy_model_save_path = policy_model_save_path
        self.target_model_load_path = target_model_load_path
        self.target_model_save_path = target_model_save_path

        # Saving episode data
        self.network_losses = list()
        self.reward_history = list()

        self.training_batch_size = training_batch_size
        self.gamma = gamma
        self.episode_cycles = episode_cycles
        self.num_episodes = num_episodes
        self.enable_player = enable_player
        self.machine_learning_setup()

    def setup(self):
        self.update_timer = 0
        self.cycle_timer = 0
        self.update_freq = UPDATE_FREQUENCY
        self.environment = \
            Environment(self.width, self.height, self.update_freq, self.graphics_enabled, self.enable_player)

    def on_update(self, delta_time):
        # Update timer
        self.update_timer += delta_time
        if self.update_timer < self.update_freq:
            return
        self.update_timer = 0

        # Episode cycle counter. Returns True when number of episode cycles has been reached.
        self.cycle_timer += 1
        if self.cycle_timer > self.episode_cycles:

            network_loss, total_reward = self.train_network()
            self.network_losses.append(network_loss)
            self.reward_history.append(total_reward)

            self.episode_timer += 1
            if self.episode_timer > self.num_episodes:
                return True
            self.setup()

        # Draw a progress bar
        i_bars = math.floor(self.cycle_timer * 20 / self.episode_cycles)
        sys.stdout.write('\r')
        sys.stdout.write("Episode %d/%d: [%-20s] %d%%    %d | %d    "
                         % (self.episode_timer,
                            self.num_episodes,
                            '=' * i_bars,
                            round(self.cycle_timer * 100 / self.episode_cycles),
                            self.cycle_timer,
                            self.episode_cycles))
        sys.stdout.flush()
        time.sleep(0.001)

        # Model returns information about all units
        state_maps = self.environment.get_state_maps()

        # For each unit, the model selects an action
        action_list = list()
        for state_map in state_maps:
            # 3D array -> 4D Tensor conversion for model compatibility
            state_map = torch.Tensor(state_map).unsqueeze(0)

            action_tensor = self.policy_model(state_map)
            action_list.append(self.policy_model.action_translate(action_tensor))

        # Action is fed back into the network
        rewards = self.environment.update(delta_time, action_list)
        next_state_maps = self.environment.get_state_maps()

        # Saves the state, action, reward and next state for training.
        for i in range(5):
            state_idx = random.randint(0, len(state_maps) - 1)
            state_t = torch.tensor(state_maps[state_idx], dtype=torch.float32, device=self.device)
            action_t = action_list[state_idx].reshape(1, )
            next_state_t = torch.tensor(next_state_maps[state_idx], dtype=torch.float32, device=self.device)
            reward_t = torch.tensor([rewards[state_idx]], dtype=torch.float32, device=self.device)
            self.memory.push(state_t, action_t, reward_t, next_state_t)

        # Return false to continue training
        return False

    def machine_learning_setup(self):
        # ====================
        # REPLAY MEMORY (DATASET)
        # ====================
        self.memory = DeepQReplay(capacity=MEMORY_CAPACITY)

        # ====================
        # MODEL
        # ====================
        if self.save_model:
            if not os.path.isdir(os.path.dirname(self.policy_model_save_path)):
                os.makedirs(os.path.dirname(self.policy_model_save_path))

        self.policy_model = DeepQNetwork_FullMap(eps_start=EPSILON_START,
                                                 eps_end=EPSILON_END,
                                                 eps_decay=EPSILON_DECAY_RATE)
        self.target_model = DeepQNetwork_FullMap(eps_start=EPSILON_START,
                                                 eps_end=EPSILON_END,
                                                 eps_decay=EPSILON_DECAY_RATE)
        self.target_model.load_state_dict(self.policy_model.state_dict())
        # Model loading
        if self.load_model:
            self.policy_model.load_state_dict(torch.load(self.policy_model_load_path))
            self.target_model.load_state_dict(torch.load(self.target_model_load_path))
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

    def train_network(self):
        # TODO: CLEAN UP COMMENTS
        # Gets a sample from the memory
        if len(self.memory) < BATCH_SIZE:
            return
        training_batch = self.memory.sample(BATCH_SIZE)
        batch = StateTransition(*zip(*training_batch))

        state_batch = torch.stack(batch.state)
        action_batch = torch.stack(batch.action)
        reward_batch = torch.stack(batch.reward)
        next_state_batch = torch.stack(batch.next_state)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_model(state_batch).gather(1, action_batch)
        print(state_action_values.item())
        with torch.no_grad():
            # Selecting the max Q trains the policy net to approximate the max
            next_state_values = self.target_model(next_state_batch).max(1)[0].unsqueeze(1)
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Huber Loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)

        sys.stdout.write("Total Reward: %.2f  Loss: %.5f  "
                         % (torch.sum(reward_batch).item(),
                            loss.item()))
        sys.stdout.flush()
        time.sleep(0.001)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_model.parameters(), 100)
        self.optimizer.step()

        # Updates the target to better match policy
        target_model_state_dict = self.target_model.state_dict()
        policy_model_state_dict = self.policy_model.state_dict()
        for key in policy_model_state_dict:
            target_model_state_dict[key] = (policy_model_state_dict[key] * TAU) \
                                           + (target_model_state_dict[key] * (1 - TAU))
        self.target_model.load_state_dict(target_model_state_dict)

        # Decays the random move tendency
        self.policy_model.random_decay_step()

        # Save the policy and target model
        if self.save_model:
            sys.stdout.write("Model Saved")
            sys.stdout.flush()
            time.sleep(0.001)
            torch.save(self.policy_model.state_dict(), self.policy_model_save_path)
            torch.save(self.target_model.state_dict(), self.target_model_save_path)
        print()

        # Return the training loss and reward achieved
        return loss.item(), torch.sum(reward_batch).item()


class UnitAI2D_Window(arcade.Window):
    def __init__(self, title, ai2d):
        super().__init__(ai2d.width, ai2d.height, title)
        arcade.set_background_color(arcade.color.AMAZON)

        self.ai2d = ai2d
        self.ai2d.graphics_enabled=True

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


def main(graphics_mode=GRAPHICS_MODE):
    random.seed(RANDOM_SEED)

    ai2d = UnitAI2D(WINDOW_WIDTH, WINDOW_HEIGHT,
                    num_episodes=NUM_EPISODES, episode_cycles=EPISODE_CYCLES,
                    training_batch_size=BATCH_SIZE, gamma=GAMMA,
                    load_model=LOAD_MODEL,
                    save_model=SAVE_MODEL,
                    policy_model_load_path=POLICY_MODEL_LOAD_PATH,
                    policy_model_save_path=POLICY_MODEL_SAVE_PATH,
                    target_model_load_path=TARGET_MODEL_LOAD_PATH,
                    target_model_save_path=TARGET_MODEL_SAVE_PATH,
                    enable_player=ENABLE_PLAYER)

    # If the display is enabled, arcade will run the game.
    # Otherwise, the update cycle will run without a window and draw
    if graphics_mode == "display":
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

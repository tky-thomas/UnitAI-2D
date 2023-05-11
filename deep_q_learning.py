import torch.nn as nn
import torch
import random


class DeepQNetwork_FullMap(nn.Module):
    def __init__(self, num_actions=5, random_action_chance=0.3, random_decay_rate=0.9):
        super().__init__()
        self.num_actions = num_actions
        self.random_action_chance = random_action_chance
        self.random_decay_rate = random_decay_rate
        self.steps = 0

        C1, C2, C3 = 3, 7, 14
        self.cnn_model = nn.Sequential(
            CNNReceptiveChunk(1, C1),
            CNNReceptiveChunk(C1, C2),
            CNNReceptiveChunk(C2, C3)
        )
        self.action_selector = nn.Sequential(
            nn.Linear(210, 100),
            nn.Linear(100, num_actions),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.cnn_model(x)
        x = x.view(x.size(dim=0), -1)
        x = self.action_selector(x)

        return x

    def action_translate(self, x):
        """
        Select an action based on the agent's prediction.
        Random chance of exploring a random move.
        """
        action = torch.argmax(x)
        if random.random() < self.random_action_chance:
            action = random.randint(0, self.num_actions - 1)
            action = torch.tensor(action)
        return action

    def random_decay_step(self):
        self.random_action_chance = self.random_action_chance * self.random_decay_rate


class CNNReceptiveChunk(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x = self.net(x)
        return x

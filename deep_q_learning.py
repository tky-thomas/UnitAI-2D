import math

import torch.nn as nn
import torch


class DeepQNetwork_FullMap(nn.Module):
    def __init__(self, num_actions=5,
                 eps_start=0.3,
                 eps_end=0.05,
                 eps_decay=0.9):
        super().__init__()
        self.num_actions = num_actions
        self.eps = 0
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.steps = 0

        C1, C2 = 8, 16
        self.cnn_model = nn.Sequential(
            nn.Conv2d(4, C1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(C1),
            nn.ReLU(),
            nn.Conv2d(C1, C2, kernel_size=3, stride=3, padding=2),
            nn.BatchNorm2d(C2),
            nn.ReLU(),
        )

        self.action_selector = nn.Sequential(
            nn.Linear(576, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
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
        x = nn.Softmax(dim=1)(x)
        action = torch.multinomial(x, num_samples=1)
        return action.item()

    def random_decay_step(self):
        self.steps += 1
        self.eps = self.eps_end + (self.eps_start * math.exp(-1 * self.steps / self.eps_decay))


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

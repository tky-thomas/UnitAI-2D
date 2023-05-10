import torch.nn as nn
import torch


class DeepQNetwork_FullMap(nn.Module):
    def __init__(self):
        super().__init__()

        C1, C2, C3 = 3, 7, 14
        self.cnn_model = nn.Sequential(
            CNNReceptiveChunk(1, C1),
            CNNReceptiveChunk(C1, C2),
            CNNReceptiveChunk(C2, C3)
        )
        self.action_selector = nn.Sequential(
            nn.Linear(C3, 100),
            nn.Linear(100, 5),
            nn.Softmax()
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


class CNNReceptiveChunk(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x = self.net(x)
        return x

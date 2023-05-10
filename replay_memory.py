from torch.utils.data import Dataset, DataLoader
from collections import namedtuple, deque

CAPACITY = 1000
StateTransition = namedtuple('StateTransition', ('state', 'action', 'reward', 'next_state'))


class DeepQReplay(Dataset):
    def __init__(self, capacity=CAPACITY):
        super().__init__()
        # Uses a queue to store the replay up to a certain point
        self.memory = deque([], maxlen=capacity)

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, idx):
        return self.memory[idx]

    def push(self, *args):
        self.memory.append(StateTransition(*args))





from collections import namedtuple

import random
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.first = nn.Linear(16, 16)
        self.bn1 = nn.BatchNorm1d(16)
        self.second = nn.Linear(16, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.final = nn.Linear(16, 4)

    def forward(self, x):
        x = F.relu(self.bn1(self.first(x)))
        x = F.relu(self.bn2(self.second(x)))
        return F.softmax(self.final(x))


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

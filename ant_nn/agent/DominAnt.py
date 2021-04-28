from .Agent import Agent
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def mish(x):
    """ Mish Activation Function """
    return x * torch.tanh(F.softplus(x))

class Brain(nn.Module):
    """ Neural Net for the ants. Uses 3 hidden layers. """

    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc2 == nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = mish(self.fc1(x))
        x = mish(self.fc2(x))
        x = self.fc2(x)
        return x


class DominAnt(Agent):
    def __init__(self, hidden_size):
        self.input = {
            "has_food": 0,
            "adjacent_food": np.zeros(5),
            "adjacent_pheromone": np.zeros(5)
        }
        input_size = 11
        output_size = 10
        self.brain = Brain(input_size, output_size, hidden_size)

    def _tensor_input(self):
        t_input = 

    def update(self):
        actions = self.brain()
        self.sense(grid)
        self.pickupFood()
        self.dropFood()
        self.depositPheromone()
        self.move(grid)

    def depositPheromone(self):
        if self.has_food:
            self.current_cell.pheromone += 1

    def move(self, grid):
        pass

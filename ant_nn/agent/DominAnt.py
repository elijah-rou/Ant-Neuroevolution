from ant_nn.agent import Agent
import torch
import torch.nn as nn
import torch.nn.functional as F

class mish(x):
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
    def __init__(self):
        self.brain = Brain()

    def update(self):
        pass

    def depositPheromone(self):
        pass

    def move(self):
        pass

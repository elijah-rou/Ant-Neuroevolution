from .Agent import Agent
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Brain(nn.Module):
    """Neural Net for the ants. Uses 3 hidden layers."""

    def __init__(self, input_size, output_size, hidden_sizes):
        super().__init__()
        self.input_fc = nn.Linear(input_size, hidden_sizes[0])
        self.hidden = []
        for i in range(len(hidden_sizes) - 1):
            self.hidden.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        self.output_fc = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        x = F.silu(self.input_fc(x))
        for h in self.hidden:
            x = F.silu(h(x))
        x = F.silu(self.output_fc(x))
        return x

    def apply_weights(self, weights):
        self.input_fc.weight.data = torch.from_numpy(weights[0]).float()
        for i, w in enumerate(weights[1:-1]):
            self.hidden[i].weight.data = torch.from_numpy(w).float()
        self.output_fc.weight.data = torch.from_numpy(weights[-1]).float()


class FetchAnt(Agent):
    MAX_TURN = np.pi / 2

    def __init__(
        self,
        hidden_sizes,
        weights,
        nest_loc=[0, 0],
        position=[0, 0],
    ):
        # Init
        super().__init__(nest_loc, position)

        # Define network input
        self.input = {
            "has_food": np.zeros(1),
            "global_angle": np.zeros(1),
            "local_angle": np.zeros(1),
        }
        input_size = 3
        output_size = 1

        # Init network and set weights
        self.brain = Brain(input_size, output_size, hidden_sizes)
        self.brain.apply_weights(weights)

    def _tensor_input(self):
        """Return a tensor from the input dict"""
        return torch.from_numpy(np.concatenate([x for x in self.input.values()]))

    def get_angle_to_nest(self):
        """returns angle from agent to nest"""
        nest_diff = self.position - (self.nest_loc + 0.5)
        theta = np.arctan2(nest_diff[1], nest_diff[0])  # angle from nest to agent
        theta = (theta + np.pi) % (2 * np.pi)  # turn around and put in 0-2pi
        return theta

    def sense(self, grid):
        """Updates current cell"""
        cell_pos = self.get_coord()  # integer coordinates of current cell
        self.current_cell = grid[cell_pos[0]][cell_pos[1]]

    def update(self, grid):
        # Update inputs
        self.sense(grid)
        self.pickupFood()
        self.dropFood()

        self.input["has_food"][0] = 1 if self.has_food else 0
        self.input["local_angle"][0] = self.orientation
        self.input["global_angle"][0] = self.get_angle_to_nest()

        # Determine actions
        actions = self.brain(self._tensor_input().float())
        self.orientation_delta = actions[0].item() * 2 * np.pi  # Orientation delta
        self.move(grid)

    def depositPheromone(self):
        self.current_cell.pheromone += self.put_pheromone

    def move(self, grid):
        # Move the approrpitae
        self.orientation += self.orientation_delta
        self.orientation %= 2 * np.pi

        next_pos = [0.0, 0.0]
        next_pos[0] = self.position[0] + self.MAX_SPEED * np.cos(self.orientation)
        next_pos[1] = self.position[1] + self.MAX_SPEED * np.sin(self.orientation)

        while not self.coord_valid(grid, next_pos):  # if walking off grid, turn around
            self.orientation = (self.orientation + np.pi / 2) % (2 * np.pi)
            next_pos[0] = self.position[0] + self.MAX_SPEED * np.cos(self.orientation)
            next_pos[1] = self.position[1] + self.MAX_SPEED * np.sin(self.orientation)

        self.position[:] = next_pos[:]

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

    def __init__(self, input_size, output_size, hidden_size, weight):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x
        return x


class DominAnt(Agent):
    PHEROMONE_MAX = 5
    MAX_TURN = np.pi / 2

    def __init__(
        self,
        hidden_size,
        weights,
        nest_loc=[0, 0],
        current_cell=None,
        sensed_cells=[None for _ in range(5)],
        position=[0, 0],
        orientation=0,
        has_food=False,
    ):
        # Init
        super().__init__(
            nest_loc, current_cell, sensed_cells, position, orientation, has_food
        )

        # Def Input
        self.input = {
            "has_food": np.array([0]),
            "adjacent_food": np.zeros(5),
            "adjacent_pheromone": np.zeros(5),
            "relative_heading": np.zeros(
                2
            ),  # orientation - tan(dy/dx), our pos - nest pos
        }
        input_size = 11
        output_size = 2

        # Init network and set weights
        self.brain = Brain(input_size, output_size, hidden_size, weights)
        self.brain.fc1.weight.data = torch.from_numpy(weights[0])
        self.brain.fc2.weight.data = torch.from_numpy(weights[1])
        self.brain.fc3.weight.data = torch.from_numpy(weights[2])

    def _tensor_input(self):
        return np.concatenate([x for x in self.input.values()])

    def sense(self, grid):
        """ Updates current and sensed cells """

        cell_pos = self.get_coord()  # integer coordinates of current cell
        self.current_cell = grid[cell_pos[0]][cell_pos[1]]

        angle_case = np.round(8 * self.orientation / (2 * np.pi)).astype(
            np.uint8
        )  # split angles 0-2pi into 8 possible cases
        # really case 0 and 8 are equivalent
        sense_coords = (
            cell_pos + self.sense_dict[angle_case]
        )  # get indices of sensed cells

        for i, coord in enumerate(sense_coords):
            if self.coord_valid(grid, coord):
                self.sensed_cells[i] = grid[coord[0]][coord[1]]
            else:
                self.sensed_cells[i] = None

        self.input["adjacent_food"] = self.sense_food()
        self.input["adjacent_pheromone"] = self.sense_pheromone()

    def sense_food(self):
        """ returns index of food in sensed cells """
        for i in self.sense_idxs:
            if self.sensed_cells[i] is not None:
                if self.sensed_cells[i].food > 0:
                    return i
        return -1

    def sense_pheromone(self):
        """ returns index of pheromone in sensed cells """
        for i in self.sense_idxs:
            if self.sensed_cells[i] is not None:
                if self.sensed_cells[i].pheromone > 0.1:
                    return i
        return -1

    def update(self):
        self.sense(grid)
        self.input["relative_heading"] = self.position - self.nest_loc
        self.pickupFood()
        self.dropFood()
        self.input["has_food"][0] = 1 if self.has_food else 0

        actions = self.brain(self._tensor_input())
        self.put_pheromone = (
            F.silu(actions[0]) * PHEROMONE_MAX
        )  # Decide to place pheromone
        self.orientation_delta = F.tanh(actions[1]) * MAX_TURN  # Orientation delta

        self.depositPheromone()
        self.move(grid)

    def depositPheromone(self):
        self.current_cell.pheromone += self.put_pheromone

    def move(self, grid):
        self.orientation += self.orientation_delta

        next_pos = [0.0, 0.0]
        next_pos[0] = self.position[0] + self.MAX_SPEED * np.cos(self.orientation)
        next_pos[1] = self.position[1] + self.MAX_SPEED * np.sin(self.orientation)

        if not self.coord_valid(grid, next_pos):  # if walking off grid, turn around
            self.orientation = (self.orientation + np.pi) % (2 * np.pi)
            next_pos[0] = self.position[0] + self.MAX_SPEED * np.cos(self.orientation)
            next_pos[1] = self.position[1] + self.MAX_SPEED * np.sin(self.orientation)

        self.position[:] = next_pos[:]

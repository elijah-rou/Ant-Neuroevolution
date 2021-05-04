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

    def __init__(self, input_size, output_size, hidden_sizes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], output_size)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = x
        return x


class DominAnt(Agent):  # IntelligAnt
    PHEROMONE_MAX = 5
    MAX_TURN = np.pi / 2

    sense_dict = {
        #                || LEFTER |  LEFT  | AHEAD |  RIGHT  | RIGHTER || RADIANS
        0: np.array([[0, 1], [1, 1], [1, 0], [1, -1], [0, -1]]),
        1: np.array([[-1, 1], [0, 1], [1, 1], [1, 0], [1, -1]]),  # pi/4
        2: np.array([[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0]]),  # pi/2
        3: np.array([[-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1]]),  # 3pi/4
        4: np.array([[0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1]]),  # pi
        5: np.array([[1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]]),  # 5pi/4
        6: np.array([[1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0]]),  # 3pi/2
        7: np.array([[1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]),  # 7pi/4
        8: np.array([[0, 1], [1, 1], [1, 0], [1, -1], [0, -1]]),  # 2pi
    }
    sense_idxs = [
        2,
        1,
        3,
        0,
        4,
    ]  # indices for use searching sensed cells in reasonable order

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
            "has_food": np.array([0]),
            "adjacent_food": np.zeros(5),
            "adjacent_pheromone": np.zeros(5),
            "relative_heading": np.zeros(2),
        }
        input_size = 13
        output_size = 2

        # Init network and set weights
        self.brain = Brain(input_size, output_size, hidden_sizes)
        self.brain.fc1.weight.data = torch.from_numpy(weights[0]).float()
        self.brain.fc2.weight.data = torch.from_numpy(weights[1]).float()
        self.brain.fc3.weight.data = torch.from_numpy(weights[2]).float()

    def _tensor_input(self):
        """ Return a tensor from the input dict """
        return torch.from_numpy(np.concatenate([x for x in self.input.values()]))

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
        result = np.zeros(5)
        for i in self.sense_idxs:
            if self.sensed_cells[i] is not None:
                if self.sensed_cells[i].food > 0:
                    result[i] = 1
        return result

    def sense_pheromone(self):
        """ returns index of pheromone in sensed cells """
        result = np.zeros(5)
        for i in self.sense_idxs:
            if self.sensed_cells[i] is not None:
                if self.sensed_cells[i].pheromone > 0.1:
                    result[i] = self.sensed_cells[i].pheromone
        return result

    def update(self, grid):
        # Update inputs
        self.sense(grid)
        self.input["relative_heading"] = self.position - self.nest_loc
        self.pickupFood()
        self.dropFood()
        self.input["has_food"][0] = 1 if self.has_food else 0

        # Determine actions
        actions = self.brain(self._tensor_input().float())
        self.put_pheromone = (
            F.silu(actions[0]).item() * self.PHEROMONE_MAX
        )  # Decide to place pheromone
        self.orientation_delta = actions[1].item() * self.MAX_TURN  # Orientation delta

        self.depositPheromone()
        self.move(grid)

    def depositPheromone(self):
        self.current_cell.pheromone += self.put_pheromone

    def move(self, grid):
        # Move the approrpitae
        self.orientation += self.orientation_delta + np.random.normal(0, 0.01)
        self.orientation %= 2 * np.pi

        next_pos = [0.0, 0.0]
        next_pos[0] = self.position[0] + self.MAX_SPEED * np.cos(self.orientation)
        next_pos[1] = self.position[1] + self.MAX_SPEED * np.sin(self.orientation)

        while not self.coord_valid(grid, next_pos):  # if walking off grid, turn around
            self.orientation = (self.orientation + np.pi / 2) % (2 * np.pi)
            next_pos[0] = self.position[0] + self.MAX_SPEED * np.cos(self.orientation)
            next_pos[1] = self.position[1] + self.MAX_SPEED * np.sin(self.orientation)

        self.position[:] = next_pos[:]

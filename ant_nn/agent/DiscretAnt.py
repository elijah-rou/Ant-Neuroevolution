import torch
import torch.nn as nn
import numpy as np

from .Agent import Agent
from .brain_util import Mish

class Brain(nn.Module):
    """Neural Net for the ants. Uses 3 hidden layers. Split branch for mean and std."""

    def __init__(self, input_size, direction_bins, pheromone_bins, hidden_sizes):
        super().__init__()
        self.input_fc = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            Mish()
        )
        self.hidden = []
        for i in range(len(hidden_sizes) - 1):
            self.hidden.append(
                nn.Sequential(
                    nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]),
                    Mish()
                )
            )

        self.output_direction = nn.Sequential(
            nn.Linear(hidden_sizes[-1], direction_bins+1),
            nn.Softmax(dim=0)
        )
        self.output_pheromone = nn.Sequential(
            nn.Linear(hidden_sizes[-1], pheromone_bins),
            nn.Softmax(dim=0)
        )
        

    def forward(self, x):
        x = self.input_fc(x)
        for h in self.hidden:                        
            x = h(x)
        x_dir = self.output_direction(x)
        x_pheromone = self.output_pheromone(x)
        x = torch.cat([x_dir, x_pheromone], 0)
        return x

    def apply_weights(self, weights):
        self.input_fc[0].weight.data = torch.from_numpy(weights[0]).float()
        for i, w in enumerate(weights[1:-2]):
            self.hidden[i][0].weight.data = torch.from_numpy(w).float()
        self.output_direction[0].weight.data = torch.from_numpy(weights[-2]).float()
        self.output_pheromone[0].weight.data = torch.from_numpy(weights[-1]).float()
        pass


class DiscretAnt(Agent):  # IntelligAnt
    PHEROMONE_MAX = 5
    MAX_RANDOM = np.pi / 8
    INPUT_SIZE = 15

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
        direction_bins,
        pheromone_bins,
        nest_loc=[0, 0],
        position=[0, 0],
    ):
        # Init
        super().__init__(nest_loc, position)

        # Define network input
        # TODO Update network input, see FetchAnt for reference
        self.input = {
            "has_food": np.array([0]),
            "adjacent_food": np.zeros(5),
            "adjacent_pheromone": np.zeros(5),
            "global_sin" : np.zeros(1),
            "global_cos" : np.zeros(1),
            "local_sin" : np.zeros(1),
            "local_cos" : np.zeros(1),
        }

        # Init network and set weights
        self.brain = Brain(self.INPUT_SIZE, direction_bins, pheromone_bins, hidden_sizes)
        self.brain.apply_weights(weights)
        self.direction_bins = direction_bins
        self.pheromone_bins = pheromone_bins

        directions = np.linspace(-np.pi/2, np.pi/2, self.direction_bins+1)
        self.directions = [(directions[i+1]+directions[i])/2 for i in range(self.direction_bins)]
        self.pheromones = np.linspace(0, self.PHEROMONE_MAX, self.pheromone_bins)

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
        """Updates current and sensed cells"""
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
        """returns index of food in sensed cells"""
        result = np.zeros(5)
        for i in self.sense_idxs:
            if self.sensed_cells[i] is not None:
                if self.sensed_cells[i].food > 0:
                    result[i] = 1
        return result

    def sense_pheromone(self):
        """returns index of pheromone in sensed cells"""
        result = np.zeros(5)
        for i in self.sense_idxs:
            if self.sensed_cells[i] is not None:
                if self.sensed_cells[i].pheromone > 0.1:
                    result[i] = self.sensed_cells[i].pheromone
        return result

    def update(self, grid):
        # Update inputs
        self.sense(grid)
        l_angle = self.orientation
        self.input["local_sin"][0] = np.sin(l_angle)
        self.input["local_cos"][0] = np.cos(l_angle)

        g_angle = self.get_angle_to_nest()
        self.input["global_sin"][0] = np.sin(g_angle)
        self.input["global_cos"][0] = np.cos(g_angle)

        self.pickupFood()
        self.dropFood()
        self.input["has_food"][0] = 1 if self.has_food else 0

        # Determine actions
        output = self.brain(self._tensor_input().float())
        direction = torch.argmax(output[:self.direction_bins+1])
        pheromone = torch.argmax(output[-self.pheromone_bins:])

        if direction < self.direction_bins:
            self.orientation_delta = self.directions[direction]
        else:
            self.orientation_delta = np.random.uniform(
            -np.pi/2, np.pi/2
        )
        self.put_pheromone = self.pheromones[pheromone]

        self.depositPheromone()
        self.move(grid)

    def depositPheromone(self):
        self.current_cell.pheromone += self.put_pheromone
        self.reward -= 2

    def move(self, grid):
        # Move the approrpitae
        # self.orientation += self.orientation_delta + np.random.normal(
        #     0, self.randomness * self.MAX_RANDOM
        # )
        self.orientation %= 2 * np.pi

        next_pos = np.array([0.0, 0.0])
        next_pos[0] = self.position[0] + self.MAX_SPEED * np.cos(self.orientation)
        next_pos[1] = self.position[1] + self.MAX_SPEED * np.sin(self.orientation)

        correction_dir = (2 * np.round(np.random.rand())) - 1  # random value either -1 or 1 to determine if turning left or right
        while not self.coord_valid(grid, next_pos):  # if walking off grid, turn
            self.orientation = (self.orientation + correction_dir * np.pi / 2) % (2 * np.pi)
            next_pos[0] = self.position[0] + self.MAX_SPEED * np.cos(self.orientation)
            next_pos[1] = self.position[1] + self.MAX_SPEED * np.sin(self.orientation)

        self.position[:] = next_pos[:]
        coords = self.get_coord()
        next_cell = grid[coords[0]][coords[1]]
        if next_cell.pheromone > 0:
            self.reward += 1
        elif next_cell.food > 0:
            self.reward += 1

        

from .Agent import Agent
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def mish(x):
    """Mish Activation Function"""
    return x * torch.tanh(F.softplus(x))


class Brain(nn.Module):
    """Neural Net for the ants. Uses 3 hidden layers."""

    # TODO implement list of hidden layers from FetchAnt
    def __init__(self, input_size, output_size, hidden_sizes):
        super().__init__()
        self.input_fc = nn.Linear(input_size, hidden_sizes[0])
        self.hidden = []
        for i in range(len(hidden_sizes) - 1):
            self.hidden.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        self.output_fc = nn.Linear(hidden_sizes[-1], output_size)

    # TODO change to silu
    def forward(self, x):
        x = torch.tanh(self.input_fc(x))
        for h in self.hidden:
            x = torch.tanh(h(x))
        x = torch.tanh(self.output_fc(x))
        return x

    # TODO Add apply_weights from FetchAnt
    def apply_weights(self, weights):
        self.input_fc.weight.data = torch.from_numpy(weights[0]).float()
        for i, w in enumerate(weights[1:-1]):
            self.hidden[i].weight.data = torch.from_numpy(w).float()
        self.output_fc.weight.data = torch.from_numpy(weights[-1]).float()


class DominAnt(Agent):  # IntelligAnt
    PHEROMONE_MAX = 5
    MAX_TURN = np.pi / 2
    MAX_RANDOM = np.pi / 8

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
        # TODO Update network input, see FetchAnt for reference
        self.input = {
            "has_food": np.array([0]),
            "adjacent_food": np.zeros(5),
            "adjacent_pheromone": np.zeros(5),
            "global_sin" : np.zeros(1),
            "global_cos" : np.zeros(1),
            "local_sin" : np.zeros(1),
            "local_cos" : np.zeros(1),
            # "global_angle": np.zeros(1),
            # "local_angle": np.zeros(1),
        }
        input_size = 15
        # input_size = 13
        # output_size = 3
        output_size = 4

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
        # TODO Fix input updates
        self.sense(grid)
        l_angle = self.orientation
        self.input["local_sin"][0] = np.sin(l_angle)
        self.input["local_cos"][0] = np.cos(l_angle)
        # self.input["local_angle"][0] = self.orientation
        g_angle = self.get_angle_to_nest()
        self.input["global_sin"][0] = np.sin(g_angle)
        self.input["global_cos"][0] = np.cos(g_angle)
        # self.input["global_angle"][0] = self.get_angle_to_nest()
        self.pickupFood()
        self.dropFood()
        self.input["has_food"][0] = 1 if self.has_food else 0

        # Determine actions
        actions = self.brain(self._tensor_input().float())
        # TODO Remove silu from this, rework network output to assume 0-1 output
        self.put_pheromone = (
            torch.sigmoid(3 * actions[0]).item()
            * self.PHEROMONE_MAX  # should set range to 0-1
        )  # Decide to place pheromone
        # self.orientation_delta = actions[1].item() * self.MAX_TURN  # Orientation delta
        # self.randomness = torch.sigmoid(
        #     3 * actions[2]
        # ).item()  # should set range to 0-1
        orientation_delta_sin = actions[1].item()
        orientation_delta_cos = actions[2].item()
        self.orientation_delta = np.arctan2(orientation_delta_sin,orientation_delta_cos) * self.MAX_TURN
        self.randomness = torch.sigmoid(
            3 * actions[3]
        ).item()  # should set range to 0-1

        self.depositPheromone()
        self.move(grid)

    def depositPheromone(self):
        self.current_cell.pheromone += self.put_pheromone

    def move(self, grid):
        # Move the approrpitae
        self.orientation += self.orientation_delta + np.random.normal(
            0, self.randomness * self.MAX_RANDOM
        )
        self.orientation %= 2 * np.pi

        next_pos = [0.0, 0.0]
        next_pos[0] = self.position[0] + self.MAX_SPEED * np.cos(self.orientation)
        next_pos[1] = self.position[1] + self.MAX_SPEED * np.sin(self.orientation)

        correction_dir = (2 * np.round(np.random.rand())) - 1  # random value either -1 or 1 to determine if turning left or right
        while not self.coord_valid(grid, next_pos):  # if walking off grid, turn
            self.orientation = (self.orientation + correction_dir * np.pi / 2) % (2 * np.pi)
            next_pos[0] = self.position[0] + self.MAX_SPEED * np.cos(self.orientation)
            next_pos[1] = self.position[1] + self.MAX_SPEED * np.sin(self.orientation)

        self.position[:] = next_pos[:]

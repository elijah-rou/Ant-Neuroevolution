import numpy as np
from ant_nn.environ.GridCell import GridCell
from ant_nn.agent.RandAnt import RandAnt
from ant_nn.agent.DeterminAnt import DeterminAnt
from ant_nn.agent.IntelligAnt import IntelligAnt
from ant_nn.agent.DiscretAnt import DiscretAnt
from ant_nn.agent.DiscretAnt2 import DiscretAnt2
from ant_nn.agent.DominAnt import DominAnt
import copy

class Environment:
    """Class representing the environment"""

    def __init__(self, config, chromosome=None, model=None):
        self.time = 0
        agent_config = config["agent"]
        
        # Setup Grid
        self.height = config.get("height", 50)
        self.width = config.get("width", 50)
        self.grid = []
        for i in range(self.height):
            self.grid.append([])
            for j in range(self.width):
                self.grid[i].append(GridCell(i, j, dissapate_coef=0.9))

        # Setup Nest
        if isinstance(config["nest_location"], str):
            if config["nest_location"] == "center":
                nest_loc = [self.height // 2, self.width // 2]
            elif config["nest_location"] == "origin":
                nest_loc = [0, 0]
        else:
            nest_loc = config["nest_location"]
        self.nest = self.grid[nest_loc[0]][nest_loc[1]]
        self.nest.is_nest = True

        # Spawn Agents
        params = agent_config.get("params")
        layer_size = params.get("hidden_layer_size")
        if chromosome and agent_config["type"] == "DominAnt":
            self.agents = [
                DominAnt(layer_size, chromosome, nest_loc=nest_loc, position=nest_loc)
                for _ in range(config["num_agents"])
            ]
        elif chromosome and agent_config["type"] == "IntelligAnt":  
            self.agents = [
                IntelligAnt(layer_size, chromosome, nest_loc=nest_loc, position=nest_loc)
                for _ in range(config["num_agents"])
            ]
        elif chromosome and agent_config["type"] == "DiscretAnt":
            d_bins = params.get("direction_bins", 7)
            p_bins = params.get("pheromone_bins", 5)
            self.agents = [
                DiscretAnt(layer_size, chromosome, d_bins, p_bins, nest_loc=nest_loc, position=nest_loc)
                for _ in range(config["num_agents"])
            ]
        elif chromosome and agent_config["type"] == "DiscretAnt2":
            d_bins = params.get("direction_bins", 7)
            self.agents = [
                DiscretAnt2(layer_size, chromosome, d_bins, nest_loc=nest_loc, position=nest_loc)
                for _ in range(config["num_agents"])
            ]
        else:
            self.agents = [
                DeterminAnt(nest_loc=nest_loc, position=nest_loc)
                for _ in range(config["num_agents"])
            ]

        if model is not None:
            for agent in self.agents:
                agent.brain = copy.deepcopy(model)
        
        self.nest_loc = nest_loc
        
        # Spawn Food
        # pick 2 sets of random row/col
        foodBoxSize = 20  # side length of square to spawn food randomly on

        spot1 = self.pick_food_loc(foodBoxSize)
        spot2 = self.pick_food_loc(foodBoxSize)

        self.spawn_food(spot1[0], spot1[1])
        self.spawn_food(spot2[0], spot2[1])

    # picks a point on a square of side length squareSize around the nest
    def pick_food_loc(self, squareSize):
        loc = [0, 0]

        sidePicker = np.random.uniform(0, 1)
        lowerBoundX = int(self.nest_loc[0] - squareSize // 2)
        upperBoundX = int(self.nest_loc[0] + squareSize // 2)
        lowerBoundY = int(self.nest_loc[1] - squareSize // 2)
        upperBoundY = int(self.nest_loc[1] + squareSize // 2)
        if sidePicker < 0.25:  # left side
            loc = [lowerBoundX, int(np.random.uniform(lowerBoundY, upperBoundY))]
        elif sidePicker < 0.5:  # right side
            loc = [upperBoundX, int(np.random.uniform(lowerBoundY, upperBoundY))]
        elif sidePicker < 0.75:  # bottom side
            loc = [int(np.random.uniform(lowerBoundX, upperBoundX)), lowerBoundY]
        else:  # top side
            loc = [int(np.random.uniform(lowerBoundX, upperBoundX)), upperBoundY]
        return loc

    def run(self, max_t=5000):
        """
        INPUT:
          max_t:
        OUTPUT:
          return number of food retrived in each time step
        """
        food_retrived = np.zeros(max_t)
        for t in range(max_t):
            self.update()
            food_retrived[t] = self.nest.food
        return food_retrived

    # def default_setup(self):
    #     nest_loc = [self.height // 2, self.width // 2]
    #     self.agents = [DeterminAnt(nest_loc=ne
    # st_loc, position=nest_loc) for _ in range(10)]
    #     # self.agents.append(DeterminAnt(nest_loc=nest_loc, position=[10,20], has_food=True))
    #     # self.agents.append(RandAnt())
    #     # Set up nest location
    #     self.spawn_food(10, 15)
    #     self.spawn_food(30, 40)

    # def dominant_setup(self):
    #     numInputs = 13
    #     numOutputs = 2
    #     hidden_size = 15
    #     nest_loc = [self.height // 2, self.width // 2]

    #     pop = Population(
    #         10, 0.1, 1, 0.1, numInputs, numOutputs, [hidden_size, hidden_size]
    #     )  # TODO: pass in real values here instead of hardcode
    #     chromosome = pop.getChromosome(0)
    #     self.agents = [DominAnt(hidden_size, chromosome, nest_loc=nest_loc, position=nest_loc) for _ in range(10)]

    #     self.spawn_food(10, 15)
    #     self.spawn_food(30, 40)

    def update(self):
        self.time += 1
        # self.drop_food()
        for grid_row in self.grid:
            for grid_cell in grid_row:
                grid_cell.update()
        for agent in self.agents:
            agent.update(self.grid)

    def drop_food(self):
        if self.time % 10 == 0:
            row = np.random.randint(self.height)
            col = np.random.randint(self.width)
            row = 10
            col = 10
            self.spawn_food(row, col)

    def spawn_food(self, row, col, r=2, amount=5):
        """
        INPUT:
          row: The row of the center of food pile
          col: The column of center of food pile
          r: radius of food pile, right now it spawn as square with side length 2r
          amount: Amount of food in each square in food pile
        OUTPUT:
          Spawn a food pile in the environment
        """
        for i in range(row - r, row + r):
            for j in range(col - r, col + r):
                if 0 <= i < self.width and 0 <= j < self.height:
                    self.grid[i][j].food += amount

        return (r+2)**2 * amount
    
    def get_agent_scores(self):
        return [(a.episode_rewards, a.episode_log_prob) for a in self.agents]

    def __str__(self):
        string = ""
        for row in self.grid:
            for cell in row:
                string += str(cell)
            string += "\n"
        return string

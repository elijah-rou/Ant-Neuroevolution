import numpy as np
from ant_nn.environ.GridCell import GridCell
from ant_nn.agent.RandAnt import RandAnt
from ant_nn.agent.DeterminAnt import DeterminAnt
from ant_nn.agent.DominAnt import DominAnt
import yaml


class Environment:
    """ Class representing the environment"""

    def __init__(self, chromosome=None):
        self.time = 0

        # Get config
        file_stream = open("config.yaml", "r")
        config = yaml.full_load(file_stream)
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
        if agent_config["type"] == "DominAnt":
            params = agent_config["params"]
            layer_size = params["hidden_layer_size"]
            self.agents = [
                DominAnt(layer_size, chromosome, nest_loc=nest_loc, position=nest_loc)
                for _ in range(config["num_agents"])
            ]
        else:
            self.agents = [
                DeterminAnt(nest_loc=nest_loc, position=nest_loc)
                for _ in range(config["num_agents"])
            ]

        # Spawn Food
        self.spawn_food(10, 15)
        self.spawn_food(30, 40)

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

<<<<<<< HEAD
    def default_setup(self):
        nest_loc = [self.height // 2, self.width // 2]
        for i in range(10):
            self.agents.append(DeterminAnt(nest_loc=nest_loc, position=nest_loc))
        # self.agents.append(DeterminAnt(nest_loc=nest_loc, position=[10,20], has_food=True))
        # self.agents.append(RandAnt())
        # Set up nest location
        self.spawn_food(10, 15)
        self.spawn_food(30, 40)

    def dominant_setup(self, chromosome = None):
        numInputs = 13
        numOutputs = 2
        hidden_size = 15
        nest_loc = [self.height // 2, self.width // 2]

        pop = Population(
            10, 0.1, 1, 0.1, numInputs, numOutputs, [hidden_size, hidden_size]
        )  # TODO: pass in real values here instead of hardcode
        if not chromosome: chromosome = pop.getChromosome(0)
=======
    # def default_setup(self):
    #     nest_loc = [self.height // 2, self.width // 2]
    #     self.agents = [DeterminAnt(nest_loc=nest_loc, position=nest_loc) for _ in range(10)]
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
>>>>>>> dev/eli

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

    def spawn_food(self, row, col, r=3, amount=1):
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

    def __str__(self):
        string = ""
        for row in self.grid:
            for cell in row:
                string += str(cell)
            string += "\n"
        return string

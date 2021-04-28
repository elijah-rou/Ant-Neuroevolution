import numpy as np
from ant_nn.environ.GridCell import GridCell
from ant_nn.agent.RandAnt import RandAnt
from ant_nn.agent.DeterminAnt import DeterminAnt
from ant_nn.agent.DominAnt import DominAnt
from ant_nn.agent.population import Population


# from GridCell import GridCell
class Environment:
    """ Class representing a cell in the environment"""

    def __init__(self, h=50, w=50, agents=[], nest=None):
        self.grid = []
        self.agents = agents
        self.time = 0
        self.height = h
        self.width = w
        self.nest = None

        for i in range(self.height):
            self.grid.append([])
            for j in range(self.width):
                self.grid[i].append(GridCell(i, j, dissapate_coef=0.9))

        if nest:
            self.nest = self.grid[nest[0]][nest[1]]
        else:
            self.nest = self.grid[h // 2][w // 2]
        self.nest.is_nest = True

    def run(self, max_t=5000):
        '''
        INPUT:
          max_t:
        OUTPUT:
          return number of food retrived in each time step
        '''
        food_retrived = np.zeros(max_t)
        for t in range(max_t):
            self.update()
            food_retrived[t] = self.nest.food
        return food_retrived

    def default_setup(self):
        nest_loc = [self.height // 2, self.width // 2]
        for i in range(10):
            self.agents.append(DeterminAnt(nest_loc=nest_loc, position=nest_loc))   
        # self.agents.append(DeterminAnt(nest_loc=nest_loc, position=[10,20], has_food=True))
        # self.agents.append(RandAnt())
        # Set up nest location
        self.spawn_food(10, 15)
        self.spawn_food(30, 40)

    def dominant_setup(self):
        numInputs = 11
        numOutputs = 2
        hidden_size = 10
        nest_loc = [self.height // 2, self.width // 2]

        pop = Population(10, .1, 1, .1, numInputs, numOutputs, [hidden_size, hidden_size]) # TODO: pass in real values here instead of hardcode
        chromosome = pop.getChromosome(0)

        for i in range(10):
            self.agents.append(DominAnt(hidden_size, chromosome, nest_loc=nest_loc, position=nest_loc))


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
        '''
        INPUT:
          row: The row of the center of food pile
          col: The column of center of food pile
          r: radius of food pile, right now it spawn as square with side length 2r
          amount: Amount of food in each square in food pile
        OUTPUT:
          Spawn a food pile in the environment
        '''
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

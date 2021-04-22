import numpy as np
from ant_nn.environ.GridCell import GridCell
from ant_nn.agent.RandAnt import RandAnt
from ant_nn.agent.DeterminAnt import DeterminAnt


# from GridCell import GridCell
class Environment:
    """ Class representing a cell in the environment"""

    def __init__(self, h=1, w=1, agents=[], nest=None):
        self.grid = []
        self.agents = agents
        self.time = 0
        self.height = h
        self.width = w
        self.nest = None

        for i in range(self.height):
            self.grid.append([])
            for j in range(self.width):
                self.grid[i].append(GridCell(i, j, dissapate_coef=0.98))
        
        if nest:
            self.nest = self.grid[nest[0]][nest[1]]
        else:
            self.nest = self.grid[h//2][w//2]
        self.nest.is_nest = True

        self.default_setup()

    def run(self):
        pass

    def default_setup(self):
        nest_loc = [self.height // 2, self.width // 2]
        for i in range(20):
            self.agents.append(DeterminAnt(nest_loc=nest_loc, position=[10,20]))
        # self.agents.append(DeterminAnt(nest_loc=nest_loc, position=[10,20], has_food=True))
        # self.agents.append(RandAnt())
        # Set up nest location
        

    def update(self):
        self.time += 1
        self.drop_food()
        for grid_row in self.grid:
            for grid_cell in grid_row:
                grid_cell.update()
        for agent in self.agents:
            agent.update(self.grid)
    
    def drop_food(self):
        if self.time % 50 == 0:
            row = np.random.randint(self.height)
            col = np.random.randint(self.width)
            self.spawn_food(row, col)

    def spawn_food(self, row, col, r=2, amount=1):
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

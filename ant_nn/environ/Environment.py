from ant_nn.environ.GridCell import GridCell
import numpy as np

# from GridCell import GridCell
class Environment:
    """ Class representing a cell in the environment"""

    def __init__(self, h=1, w=1, agents=[]):
        self.grid = []
        self.agents = agents
        self.time = 0
        self.height = h
        self.width = w

        for i in range(self.height):
            self.grid.append([])
            for j in range(self.width):
                self.grid[i].append(GridCell(i, j))

        # print(self.__str__())

    def run(self):
        pass

    def default_setup(self):
        pass

    def update(self):
        self.time += 1
        self.drop_food()
        for grid_row in self.grid:
            for grid_cell in grid_row:
                grid_cell.update()
        for agent in self.agents:
            agent.update(self)
    
    def drop_food(self):
        if self.time % 10 == 0:
            row = np.random.randint(self.height)
            col = np.random.randint(self.width)
            self.spawn_food(row, col)

    def spawn_food(self, row, col, r=3, amount=1):
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

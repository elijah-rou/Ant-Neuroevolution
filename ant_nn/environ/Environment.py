from ant_nn.environ.GridCell import GridCell
# from GridCell import GridCell
class Environment:
    """ Class representing a cell in the environment"""
    def __init__(
        self,
        h = 1,
        w = 1,
        agents = []

    ):
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
        self.grid[2][1].food += 2
        for i in range(2,5):
            self.grid[3][i].active = False
        for i in range(1,9):
            self.grid[6][i].pheromone = 1 * 0.9 ** i

    def update(self):
        for grid_row in self.grid:
            for grid_cell in grid_row:
                grid_cell.update()
        for agent in self.agents:
            agent.update(self)

    
    def __str__(self):
        string = ""
        for row in self.grid:
            for cell in row:
                string += str(cell)
            string += '\n'
        return string

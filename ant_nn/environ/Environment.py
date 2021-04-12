from GridCell import GridCell

class Environment:
    """ Class representing a cell in the environment"""
    def __init__(
        self,
        h = 1,
        w = 1,
        agents = None

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
        
        print(self.__str__())
    
    def run(self):
        pass

    def update(self):
        for grid_cell in self.grid:
            grid_cell.update()
        for agent in self.agents:
            agent.update(self.grid)

    
    def __str__(self):
        string = ""
        for row in self.grid:
            for cell in row:
                string += str(cell)
            string += '\n'
        return string

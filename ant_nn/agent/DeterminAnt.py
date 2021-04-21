import Agent

class DeterminAnt(Agent):
    def __init__(self):
        super.__init__()

    def update(self, current_cell, sensed_cells):
        self.current_cell = current_cell
        self.sensed_cells = sensed_cells
        
        self.pickupFood()
        self.dropFood()
        self.depositPheromone()
        self.move()

    def depositPheromone(self):
        if (self.has_food):
            self.current_cell.pheromone += 1

    def move(self):
        if (self.has_food):  # head straight to colony w/ food
            self.orientation = self.position[1] + np.pi
            self.velocity = MAX_VEL
        else:

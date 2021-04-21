import Agent

class DeterminAnt(Agent):
    def __init__(self):
        super.__init__()

    def update(self):
        pass

    def depositPheromone(self):
        pass

    def move(self):
        if(self.has_food):
            self.orientation = self.position[1] + np.pi
            self.velocity = MAX_VEL
        else:

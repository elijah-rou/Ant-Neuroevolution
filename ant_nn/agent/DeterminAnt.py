import Agent
import numpy as np

class DeterminAnt(Agent):
    def __init__(self):
        super.__init__()

    def update(self, env):
        self.sense(env)

        self.pickupFood()
        self.dropFood()
        self.depositPheromone()
        self.move()

    def self.sense(env):
        cell_pos = self.get_coord()  # integer coordinates of current cell
        angle_case = np.round(8 * self.orientation / (2*np.pi)).astype(np.uint8)  # split angles 0-2pi into 8 possible cases
                                                                                  # really case 0 and 8 are equivalent
        sense_dict = {
            #   || LEFTER |  LEFT  | AHEAD |  RIGHT  | RIGHTER || RADIANS
            8 :  [[ 0, 1], [ 1, 1], [ 1, 0], [ 1,-1], [ 0,-1]],  # 0
            0 :  [[ 0, 1], [ 1, 1], [ 1, 0], [ 1,-1], [ 0,-1]],
            1 :  [[-1, 1], [ 0, 1], [ 1, 1], [ 1, 0], [ 1,-1]],  # pi/4
            2 :  [[-1, 0], [-1, 1], [ 0, 1], [ 1, 1], [ 1, 0]],  # pi/2
            3 :  [[-1,-1], [-1, 0], [-1, 1], [ 0, 1], [ 1, 1]],  # 3pi/4
            4 :  [[ 0,-1], [-1,-1], [-1, 0], [-1, 1], [ 0, 1]],  # pi
            5 :  [[ 1,-1], [ 0,-1], [-1,-1], [-1, 0], [-1, 1]],  # 5pi/4
            6 :  [[ 1, 0], [ 1,-1], [ 0,-1], [-1,-1], [-1, 0]],  # 3pi/2
            7 :  [[ 1, 1], [ 1, 0], [ 1,-1], [ 0,-1], [-1,-1]],  # 7pi/4
        }

        sens_coords = sense_dict[angle_case]  # get indices of sensed cells
        self.sensed_cells = env[sense_coords[:,0], sens_coords[:,1]]
        

    def depositPheromone(self):
        if self.has_food:
            self.current_cell.pheromone += 1

    def move(self):
        if self.has_food:  # head straight to colony w/ food
            theta = np.arctan(
                self.position[1] / self.position[0]
            )  # polar angle from nest
            self.orientation = (theta + np.pi) % (2 * np.pi)
            self.speed = MAX_SPEED
        else:
            self.orientation = (self.orientation + np.random.normal(0, 0.5)) % (2 * np.pi)
            self.speed = MAX_SPEED
        self.position[0] = self.speed * cos(self.orientation)
        self.position[1] = self.speed * sin(self.orientation)

from .Agent import Agent
import numpy as np

class DeterminAnt(Agent):

    # dictionary for looking up indices of sensed cells
    sense_dict = {
    #                || LEFTER |  LEFT  | AHEAD |  RIGHT  | RIGHTER || RADIANS
        8 :  np.array([[ 0, 1], [ 1, 1], [ 1, 0], [ 1,-1], [ 0,-1]]),  # 0
        0 :  np.array([[ 0, 1], [ 1, 1], [ 1, 0], [ 1,-1], [ 0,-1]]),
        1 :  np.array([[-1, 1], [ 0, 1], [ 1, 1], [ 1, 0], [ 1,-1]]),  # pi/4
        2 :  np.array([[-1, 0], [-1, 1], [ 0, 1], [ 1, 1], [ 1, 0]]),  # pi/2
        3 :  np.array([[-1,-1], [-1, 0], [-1, 1], [ 0, 1], [ 1, 1]]),  # 3pi/4
        4 :  np.array([[ 0,-1], [-1,-1], [-1, 0], [-1, 1], [ 0, 1]]),  # pi
        5 :  np.array([[ 1,-1], [ 0,-1], [-1,-1], [-1, 0], [-1, 1]]),  # 5pi/4
        6 :  np.array([[ 1, 0], [ 1,-1], [ 0,-1], [-1,-1], [-1, 0]]),  # 3pi/2
        7 :  np.array([[ 1, 1], [ 1, 0], [ 1,-1], [ 0,-1], [-1,-1]]),  # 7pi/4
    }

    def __init__(
        self, 
        nest_loc = [0,0],
        current_cell=None, 
        sensed_cells=[None for i in range(5)], 
        position= [0,0], 
        orientation=0, 
        velocity=np.array((0,0)),
    ):
        super().__init__(nest_loc, current_cell, sensed_cells, position, orientation, velocity)

    def update(self, grid):
        grid
        self.sense(grid)
        if(self.current_cell.is_nest):
            print('IN NEST')

        self.pickupFood()
        self.dropFood()
        self.depositPheromone()
        self.move(grid)

    def sense(self,grid):
        """ Updates current and sensed cells """

        cell_pos = self.get_coord()  # integer coordinates of current cell
        self.current_cell = grid[cell_pos[0]][cell_pos[1]]

        angle_case = np.round(8 * self.orientation / (2*np.pi)).astype(np.uint8)  # split angles 0-2pi into 8 possible cases
                                                                                  # really case 0 and 8 are equivalent
        sense_coords = cell_pos + self.sense_dict[angle_case]  # get indices of sensed cells

        for i,coord in enumerate(sense_coords):
            if (self.coord_valid(grid,coord)):
                self.sensed_cells[i] = grid[coord[0]][coord[1]]
            else:
                self.sensed_cells[i] = None

        

    def depositPheromone(self):
        if self.has_food:
            self.current_cell.pheromone += 1

    def move(self, grid):
        if self.has_food:  # head straight to colony w/ food
            nest_diff = self.position - (self.nest_loc + 0.5)
            theta = np.arctan2(nest_diff[1], nest_diff[0])
            self.orientation = (theta + np.pi) % (2 * np.pi)
            self.speed = self.MAX_SPEED
        else:
            # random walk
            self.orientation = (self.orientation + np.random.normal(0, 0.5)) % (2 * np.pi)
            self.speed = self.MAX_SPEED

        next_pos = [0.,0.]
        next_pos[0] = self.position[0] + self.speed * np.cos(self.orientation)
        next_pos[1] = self.position[1] + self.speed * np.sin(self.orientation)

        if not self.coord_valid(grid,next_pos):  # if walking off grid, turn around
            self.orientation = (self.orientation + np.pi) % (2 * np.pi)
            next_pos[0] = self.position[0] + self.speed * np.cos(self.orientation)
            next_pos[1] = self.position[1] + self.speed * np.sin(self.orientation)

        self.position[:] = next_pos[:]

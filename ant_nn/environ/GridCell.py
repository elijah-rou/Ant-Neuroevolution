class GridCell:
    """ Class representing a cell in the environment"""
    def __init__(
        self,
        init_params: dict
    ):
        self.pheromone = 0
        self.active = not init_params["is-wall"]
        if self.active:
            self.food = init_params["food"]
        else:
            self.food = 0
            

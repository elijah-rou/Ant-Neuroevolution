class Environment:
    """ Class representing a cell in the environment"""
    def __init__(
        self,
        init_params: dict
    ):
        self.grid = []
        self.agents = init_params["agents"]
        self.time = 0
    
    def run(self):
        pass

    def update(self):
        for agent in self.agents:
            agent.update(self.grid)

            

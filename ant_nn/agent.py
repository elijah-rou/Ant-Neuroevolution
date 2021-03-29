class Agent:
    """Class representing the ant agent"""
    def __init__(self, agent_cls):
        self.brain = agent_cls()

    def update(self):
        pass
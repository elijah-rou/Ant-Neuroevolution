import Agent

class DeterminAnt(Agent):
    def __init__(self):
        super.__init__()

        @abstractmethod
    def update(self):
        """ Update the Agent's state """
        raise NotImplementedError

    @abstractmethod
    def depositPheromone(self):
        """ Decide whether to drop pheromone """
        raise NotImplementedError

    @abstractmethod
    def move(self):
        """ Decide a direction to move """
        raise NotImplementedError

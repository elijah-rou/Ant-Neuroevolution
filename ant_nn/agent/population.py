# Ant Neuroevolution Project
# Population class
# Author: Russell Bingham, Eli Roussos
# Date: 3/31/21
from .DiscretAnt import DiscretAnt
import numpy as np
import random
from .IntelligAnt import IntelligAnt
from .DominAnt import DominAnt

class Population:

    """Class representing the GA chromosome population"""

    # assuming the following rough usage syntax:
    # pop = Population(...params...)
    # for e in range(numEpochs):
    #     for i in range(pop.size()):
    #         currentChromosome = pop.getChromosome(i)
    #         ... turn chromosome into network ...
    #         ... run sim ...
    #         pop.setScore(i, finalScore)
    #     pop.makeBabies()

    def __init__(
        self,
        popSize,
        mutationRate,
        mutationStrength,
        keepThresh,
        agentConfig,
        initFromFile=False,
        filename=None,
    ):
        self.popSize = popSize  # number of chromosomes
        self.mutationRate = mutationRate  # probability of a given weight getting mutated (keep low) (i.e. 0.1)
        self.maxMutationStrength = (
            mutationStrength  # variance of gaussian mutation function (needs testing)
        )
        self.clampRange = [-2, 2]  # range of allowable scores
        self.keepThresh = keepThresh  # what percentage of best chromosomes to keep unchanged each epoch (try 0.1)
        self.crossover = False  # enable crossover, not implemented yet
        if initFromFile:
            import pickle

            pickle_off = open(filename, "rb")
            temp = pickle.load(pickle_off)
            level = temp[2]
            # print(level[0])
            # for i in range(len(level)):
            #     level[i] = np.asarray(level[i])
            # print(level[0].shape)
            self.chromosomes = level
        else:
            self.chromosomes = self.initializePop(
                agentConfig
            )  # list of weights
        self.scores = np.zeros(popSize)  # list of scores

        self.mutationStrength = 0  # temp

        self.maxScore = 160  # represents the target score - WARNING - if scores go above this training stops

    # makes self.chromosomes
    def initializePop(self, agentConfig):
        agentType = agentConfig["type"]
        params = agentConfig["params"]

        layerSizes = params["hidden_layer_size"]
        layerShapes = [
            (layerSizes[i+1], layerSizes[i]) for i in range(len(layerSizes) -1)
        ]
        if agentType == "DominAnt":
            layerShapes = [(layerSizes[0], DominAnt.INPUT_SIZE)] + layerShapes
            layerShapes += [(DominAnt.OUTPUT_SIZE, layerSizes[-1])]
        elif agentType == "IntelligAnt":
            layerShapes = [(layerSizes[0], DiscretAnt.INPUT_SIZE)] + layerShapes
            layerShapes += 2*[(IntelligAnt.OUTPUT_SIZE, layerSizes[-1])]
        elif agentType == "DiscretAnt":
            d_bins = params["direction_bins"]
            p_bins = params["pheromone_bins"]
            layerShapes = [(layerSizes[0], DiscretAnt.INPUT_SIZE)] + layerShapes
            layerShapes += [(d_bins+1, layerSizes[-1])]
            layerShapes += [(p_bins, layerSizes[-1])]
        
        popArray = []
        for _ in range(self.popSize):
            popArray += [self.makeChromosome(layerShapes)]
        return popArray

    # makes a single chromosome, returns it
    # dimensionality will be a list of numpy arrays, inner dims given by params
    def makeChromosome(self, layerShapes, randomCenter=0, randomWidth=1):
        chromosome = []
        for shape in layerShapes:
            chromosome += [
                randomWidth
                    * (
                        np.random.rand(shape[0], shape[1])
                        - (0.5 - randomCenter)
                    )
            ]

        return chromosome

    # bound weights to range
    def clampChromosome(self, chromosome):
        for i in range(len(chromosome)):
            chromosome[i] = np.where(
                chromosome[i] > self.clampRange[0], chromosome[i], self.clampRange[0]
            )
            chromosome[i] = np.where(
                chromosome[i] < self.clampRange[1], chromosome[i], self.clampRange[1]
            )
        return chromosome

    # assumes scores have already been set by sim
    # resamples pop and generates new individuals by mutation
    # TODO: add crossover routine at the end to cross-pollinate new individuals
    def makeBabies(self):
        newGen = []
        best_score = np.max(self.scores)
        keepCutoff = np.quantile(
            self.scores, (1.0 - self.keepThresh), interpolation="lower"
        )
        numKeeps = int(self.popSize * self.keepThresh)
        

        counter = 0
        # carry over the best individuals
        for i in range(self.popSize):
            if self.scores[i] >= keepCutoff:
                newGen += [self.chromosomes[i]]
                counter += 1
                if counter >= numKeeps:
                    break

        if counter + 1 < numKeeps:
            print("flag")

        # mutate new individuals
        for i in range(self.popSize - numKeeps):
            mutant = newGen[int(numKeeps * random.random())].copy()
            mutant = self.mutate(mutant, best_score)
            newGen += [mutant]

        self.chromosomes = newGen

        # TODO: put crossover routine here
        # TODO: add more sexual innuendos to this method

    # takes in chromosome, randomly mutates it according to stored params
    def mutate(self, chromosome, score):
        # mutate more if score is low
        self.mutationStrength = self.maxMutationStrength * (1 - (score / self.maxScore))
        # loop over layers
        for i in range(len(chromosome)):
            # loop over weights
            for j in range(chromosome[i].shape[0]):
                for k in range(chromosome[i].shape[1]):
                    if (
                        random.random() < self.mutationRate
                    ):  # only mutate a gene w some small prob
                        chromosome[i][j][k] += np.random.normal(0, self.mutationStrength)

        chromosome = self.clampChromosome(chromosome)
        return chromosome

    def setScore(self, index, score):
        self.scores[index] = score

    # return a full chromosome by index
    def getChromosome(self, index):
        return self.chromosomes[index]

    # return number of stored chromosomes
    # thought is to loop through
    def size(self):
        return self.popSize


# testing code - uncomment to see functionality
# testPop = Population(10, .8, 1, .1, 4, 2, [3])
# for i in range(50):
#     for j in range(testPop.size()):
#         testPop.setScore(j, 100*random.random())
#     testPop.makeBabies()
# print(testPop.getChromosome(1))

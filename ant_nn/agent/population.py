# Ant Neuroevolution Project
# Population class
# Author: Russell Bingham
# Date: 3/31/21
import numpy as np
import random


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
        numInputs,
        numOutputs,
        layerSizes,
    ):
        self.popSize = popSize  # number of chromosomes
        self.maxMutationRate = mutationRate  # probability of a given weight getting mutated (keep low) (i.e. 0.1)
        self.mutationStrength = (
            mutationStrength  # variance of gaussian mutation function (needs testing)
        )
        self.clampRange = [-1, 1] # range of allowable scores
        self.keepThresh = keepThresh  # what percentage of best chromosomes to keep unchanged each epoch (try 0.1)
        self.crossover = False  # enable crossover, not implemented yet
        self.chromosomes = self.initializePop(
            numInputs, numOutputs, layerSizes
        )  # list of weights
        self.scores = np.zeros(popSize)  # list of scores
        
        self.mutationRate = 0 # temp

        self.maxScore = 320 #WARNING ----- this is hard coded and needs to be updated if food ICs change

    # makes self.chromosomes
    def initializePop(self, numInputs, numOutputs, layerSizes):
        popArray = []
        for i in range(self.popSize):
            popArray += [self.makeChromosome(numInputs, numOutputs, layerSizes)]
        return popArray

    # makes a single chromosome, returns it
    # dimensionality will be a list of numpy arrays, inner dims given by params
    def makeChromosome(self, numInputs, numOutputs, hiddenSizes):
        chromosome = []

        # TODO: optimize these coefficients/make them not hard-coded
        randomnessCenter = 0  # center of initialization range
        randomnessWidth = .1  # width of initialization range

        for i in range(len(hiddenSizes) + 1):
            if i == 0:  # first layer
                chromosome += [
                    randomnessWidth
                    * (
                        np.random.rand(hiddenSizes[0], numInputs)
                        - (0.5 - randomnessCenter)
                    )
                ]
            elif i == len(hiddenSizes):  # last layer
                chromosome += [
                    randomnessWidth
                    * (
                        np.random.rand(numOutputs, hiddenSizes[-1])
                        - (0.5 - randomnessCenter)
                    )
                ]
            else:  # middle layers
                chromosome += [
                    randomnessWidth
                    * (
                        np.random.rand(hiddenSizes[i], hiddenSizes[i - 1])
                        - (0.5 - randomnessCenter)
                    )
                ]
        chromosome = self.clampChromosome(chromosome)
        return chromosome

    # bound weights to range
    def clampChromosome(self, chromosome):
        for i in range(len(chromosome)): 
            chromosome[i] = np.where(chromosome[i] > self.clampRange[0], chromosome[i], self.clampRange[0])
            chromosome[i] = np.where(chromosome[i] < self.clampRange[1], chromosome[i], self.clampRange[1])
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
            mutant = newGen[int(numKeeps * random.random())]
            mutant = self.mutate(mutant, best_score)
            newGen += [mutant]

        # TODO: put crossover routine here
        # TODO: add more sexual innuendos to this method

    # takes in chromosome, randomly mutates it according to stored params
    def mutate(self, chromosome, score):
        # mutate more if score is low
        self.mutationRate = self.maxMutationRate - (score/self.maxScore)
        # loop over layers
        for i in range(len(chromosome)):
            # loop over weights
            for j in range(chromosome[i].shape[0]):
                for k in range(chromosome[i].shape[1]):
                    if (
                        random.random() < self.mutationRate
                    ):  # only mutate a gene w some small prob
                        chromosome[i][j][k] = random.uniform(self.clampRange[0], self.clampRange[1])

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
# testPop = Population(100, .1, 1, .1, 5, 5, [6, 7])
# for i in range(testPop.size()):
#     testPop.setScore(i, 100*random.random())
# for i in range(50):
#     testPop.makeBabies()
# print(testPop.getChromosome(99))

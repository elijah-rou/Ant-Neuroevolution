from ant_nn.simulation import Simulation
import dill


def main():
    sim = Simulation()
    chromosomes, scores, final_pop, food = sim.run()

    file = open("results.pkl", "wb")
    dill.dump([chromosomes, scores, final_pop, food], file)
    file.close()
    print("done")


if __name__ == "__main__":
    main()

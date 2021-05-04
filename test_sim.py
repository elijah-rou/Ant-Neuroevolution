from ant_nn.simulation import Simulation
import dill

def main():
    sim = Simulation()
    chromosomes, scores = sim.run()

    file = open("results.pkl", "wb")
    dill.dump([chromosomes, scores], file)
    file.close()
    print(scores)
    print("done")


if __name__ == "__main__":
    main()

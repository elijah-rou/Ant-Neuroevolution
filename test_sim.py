from ant_nn.simulation import Simulation
import dill

def main():
    sim = Simulation()
    best_chromosomes, best_scores = sim.run()

    file = open("results.pkl", "wb")
    dill.dump([best_scores, best_chromosomes], file)
    file.close()
    print(best_scores)
    print("done")


if __name__ == "__main__":
    main()

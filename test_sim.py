from ant_nn.simulation import Simulation


def main():
    sim = Simulation(setup="domi")
    sim.sample_experiment()
    print("done")


if __name__ == "__main__":
    main()

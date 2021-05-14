  
#!/usr/bin/env python
#=========================================================================
# run-sim [options]
#=========================================================================
#
#  -c --config <path>  set path to config file (.yaml)
#  -r --result <path>  set path where results will be written (.pkl)
#
# Author : Darren Midkiff
# Date   : May 13, 2021
#

import sys
import getopt
from ant_nn.simulation import Simulation
import dill


def main(cmdline_opts):
    try:
        short_opts = 'c:r:d'
        long_opts = ['config=', 'result=', 'degen-epoch=', 'degen-score=']
        optlist, args = getopt.getopt(cmdline_opts, short_opts, long_opts)
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)  # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    # default paths
    config_path = "config.yaml"
    result_path = "result.yaml"

    # default degen settings
    degen_epoch = None
    degen_score = 10

    # parse args
    for opt,arg in optlist:
        if opt in ("-c", "--config"):
            config_path = arg
        if opt in ("-r", "--result"):
            result_path = arg
        if opt == '-d':
            degen_epoch = 50
        if opt == "--degen-epoch":
            degen_epoch = int(arg)
        if opt == "--degen-score":
            degen_score = int(arg)

    print("config file:", config_path)
    print("result file:", result_path)
    print()

    sim = Simulation(config_path=config_path)
    chromosomes, scores, final_pop, food = sim.run(degen_epoch=degen_epoch, degen_score=degen_score)

    file = open(result_path, "wb")
    dill.dump([chromosomes, scores, final_pop, food], file)
    file.close()
    print("done")


if __name__ == "__main__":
    cmdline_opts = sys.argv[1:]
    main( cmdline_opts )
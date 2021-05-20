import sys
import getopt

def main( cmdline_opts ):
    try:
        long_opts = ['config=', 'result=']
        optlist, args = getopt.getopt(cmdline_opts, 'c:f:', long_opts)
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)  # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    print(optlist)

    config_path = "config.yaml"
    result_path = "result.yaml"

    for opt,arg in optlist:
        if opt in ("-c", "--config"):
            config_path = arg
        if opt in ("-r", "--result"):
            result_path = arg

    print("config:",config_path)
    print("result:",result_path)


if __name__ == "__main__":
    cmdline_opts = sys.argv[1:]
    main( cmdline_opts )
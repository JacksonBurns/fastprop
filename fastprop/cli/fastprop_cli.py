import argparse
import yaml
import sys
from importlib.metadata import version

from fastprop.utils import validate_config


def main():
    parser = argparse.ArgumentParser(description="fastprop command line interface - try 'fastprop subcommand --help'")

    parser.add_argument("-v", "--version", help="print version and exit", action="store_true")

    subparsers = parser.add_subparsers(dest="subcommand")

    train_subparser = subparsers.add_parser("train")

    config_file_arg = train_subparser.add_argument("config_file", nargs="?", help="YAML configuration file")
    train_subparser.add_argument("-i", "--interaction-layers", type=int, help="number of interactions layers")
    train_subparser.add_argument("-f", "--fnn-layers", type=int, help="number of fnn layers")

    predict_subparser = subparsers.add_parser("predict")
    predict_subparser.add_argument("-c", "--checkpoint", help="checkpoint file for predictions")
    predict_subparser.add_argument("-s", "--smiles", help="SMILES string for prediction")
    predict_subparser.add_argument("-o", "--output", required=False, help="output file for predictions")
    # by default, will create a new file with the same name as the input file plus a timestamp

    # no args provided - print the help text
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()
    if args.version:
        print(version("fastprop"))
        exit(0)

    # cast to a dict
    args = vars(args)
    subcommand = args.pop("subcommand")
    args.pop("version")
    match subcommand:
        case "train":
            # exit with help if no args given
            if not sum(map(lambda i: i is not None, args.values())):
                train_subparser.print_help()
                exit(0)

            training_default = {"interaction_layers": 2, "fnn_layers": 3}

            if args["config_file"] is not None:
                if sum(map(lambda i: i is not None, args.values())) > 1:
                    raise argparse.ArgumentError(
                        message="Cannot specify config_file with other command line arguments.",
                        argument=config_file_arg,
                    )
                with open(args["config_file"], "r") as f:
                    cfg = yaml.safe_load(f)
                    training_default.update(cfg)
            else:
                training_default.update({k: v for k, v in args.items() if v is not None})

            print(training_default)
            # validate this dictionary, i.e. layer counts are positive, dropout rates reasonable, etc.
        case "predict":
            ...


if __name__ == "__main__":
    main()

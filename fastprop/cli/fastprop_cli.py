import argparse
import json
import yaml
import sys
from importlib.metadata import version

from fastprop import DEFAULT_TRAINING_CONFIG
from fastprop.utils import validate_config
from fastprop import train_fastprop


def main():
    parser = argparse.ArgumentParser(description="fastprop command line interface - try 'fastprop subcommand --help'")

    parser.add_argument("-v", "--version", help="print version and exit", action="store_true")

    subparsers = parser.add_subparsers(dest="subcommand")

    train_subparser = subparsers.add_parser("train")
    train_subparser.add_argument("config_file", nargs="?", help="YAML configuration file")
    train_subparser.add_argument("-od", "--output-directory", help="directory for fastprop output")
    # featurization
    train_subparser.add_argument("-if", "--input-file", help="csv of SMILES and targets")
    train_subparser.add_argument("-tc", "--target-columns", nargs="+", help="column name(s) for target(s)")
    train_subparser.add_argument("-sc", "--smiles-column", help="column name for SMILES")
    train_subparser.add_argument("-d", "--descriptors", help="descriptors to calculate (one of all, optimized, smallest, or search)")
    train_subparser.add_argument("-ec", "--enable-cache", type=bool, help="allow saving and loading of cached descriptors")
    train_subparser.add_argument("-p", "--precomputed", help="precomputed descriptors from fastprop or mordred")

    # preprocessing
    train_subparser.add_argument("-r", "--rescaling", type=bool, help="rescale descriptors between 0 and 1 (default to True)")
    train_subparser.add_argument("-zvd", "--zero-variance-drop", type=bool, help="drop zero variance descriptors (defaults to True)")
    train_subparser.add_argument("-cd", "--colinear-drop", type=bool, help="drop colinear descriptors (defaults to False)")

    # training
    train_subparser.add_argument("-il", "--interaction-layers", type=int, help="number of interactions layers")
    train_subparser.add_argument("-dr", "--dropout-rate", type=float, help="dropout rate for interaction layers")
    train_subparser.add_argument("-fl", "--fnn-layers", type=int, help="number of fnn layers")
    train_subparser.add_argument("-lr", "--learning-rate", type=float, help="learning rate")
    train_subparser.add_argument("-bs", "--batch-size", type=int, help="batch size")
    train_subparser.add_argument("-ne", "--number-epochs", type=int, help="number of epochs")
    train_subparser.add_argument("-pt", "--problem-type", help="problem type (regression, classification)")

    predict_subparser = subparsers.add_parser("predict")
    predict_subparser.add_argument("-c", "--checkpoint", required=True, help="checkpoint file for predictions")
    input_group = predict_subparser.add_mutually_exclusive_group()
    input_group.add_argument("-s", "--smiles", help="SMILES string for prediction")
    input_group.add_argument("-i", "--input-file", help="file containing SMILES strings")
    predict_subparser.add_argument("-o", "--output", required=False, help="output file for predictions (defaults to stdout)")

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
            training_default = dict(DEFAULT_TRAINING_CONFIG)

            # exit with help if no args given
            if not sum(map(lambda i: i is not None, args.values())):
                train_subparser.print_help()
                exit(0)

            if args["config_file"] is not None:
                if sum(map(lambda i: i is not None, args.values())) > 1:
                    raise parser.error("Cannot specify config_file with other command line arguments.")
                with open(args["config_file"], "r") as f:
                    cfg = yaml.safe_load(f)
                    cfg["target_columns"] = cfg["target_columns"].split(" ")
                    training_default.update(cfg)
            else:
                # cannot specify both precomputed and descriptors or enable/cache
                training_default.update({k: v for k, v in args.items() if v is not None})

            print("training parameters:\n", json.dumps(training_default, indent=4))
            # validate this dictionary, i.e. layer counts are positive, dropout rates reasonable, etc.
            train_fastprop(**training_default)
        case "predict":
            if args["smiles"] is None and args["input_file"] is None:
                raise parser.error("One of -i/--input-file or -s/--smiles must be provided.")
            print(args)

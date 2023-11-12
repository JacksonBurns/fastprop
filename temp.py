import argparse
import yaml


parser = argparse.ArgumentParser(description="fastprop command line interface")

subparsers = parser.add_subparsers(dest="subcommand")

train_subparser = subparsers.add_parser("train")

train_subparser.add_argument("-c", "--config-file", help="YAML configuration file")
train_subparser.add_argument("-i", "--interaction-layers", type=int, help="number of interactions layers")
train_subparser.add_argument("-f", "--fnn-layers", type=int, help="number of fnn layers")

predict_subparser = subparsers.add_parser("predict")
predict_subparser.add_argument("-c", "--checkpoint", help="checkpoint file for predictions")
predict_subparser.add_argument("-s", "--smiles", help="SMILES string for prediction")
predict_subparser.add_argument("-o", "--output", required=False, help="output file for predictions")
# by default, will create a new file with the same name as the input file plus a timestamp

args = parser.parse_args()
if args.subcommand == "train":
    # cast to a dict
    args = vars(args)
    args.pop("subcommand")

    # exit if no args given
    if not sum(map(lambda i: i is not None, args.values())):
        raise RuntimeError("No command line options given, see fastprop train --help")

    training_default = {"interaction_layers": 2, "fnn_layers": 3}

    if args["config_file"] is not None:
        if sum(map(lambda i: i is not None, args.values())) > 1:
            raise RuntimeError("Cannot specify -c/--config-file with other command line arguments.")
        with open(args["config_file"], "r") as f:
            cfg = yaml.safe_load(f)
            training_default.update(cfg)
    else:
        training_default.update({k: v for k, v in args.items() if v is not None})

    print(training_default)
    # validate input
elif args.subcommand == "predict":
    ...

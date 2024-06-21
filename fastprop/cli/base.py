import argparse
import datetime
import sys
from importlib.metadata import version
from time import perf_counter

import yaml

from fastprop import DEFAULT_TRAINING_CONFIG
from fastprop.defaults import init_logger

from .predict import predict_fastprop
from .shap import shap_fastprop
from .train import train_fastprop

logger = init_logger(__name__)


def main():
    cli_start = perf_counter()
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
    train_subparser.add_argument("-ds", "--descriptor-set", help="descriptors to calculate (one of all, optimized, or debug)")
    train_subparser.add_argument("-ec", "--enable-cache", type=bool, help="allow saving and loading of cached descriptors")
    train_subparser.add_argument("-p", "--precomputed", help="precomputed descriptors from fastprop or mordred")

    # training
    train_subparser.add_argument("-op", "--optimize", action="store_true", help="run hyperparameter optimization", default=False)
    train_subparser.add_argument("-hs", "--hidden-size", type=int, help="hidden size of fnn layers")
    train_subparser.add_argument("-fl", "--fnn-layers", type=int, help="number of fnn layers")
    train_subparser.add_argument("-ci", "--clamp-input", action="store_true", help="clamp inputs to +/-3", default=False)
    train_subparser.add_argument("-lr", "--learning-rate", type=float, help="learning rate")
    train_subparser.add_argument("-bs", "--batch-size", type=int, help="batch size")
    train_subparser.add_argument("-ne", "--number-epochs", type=int, help="number of epochs")
    train_subparser.add_argument("-nr", "--number-repeats", type=int, help="number of repeats")
    train_subparser.add_argument("-pt", "--problem-type", help="problem type (regression or a type of classification)")
    train_subparser.add_argument("-ns", "--train-size", type=float, help="train size")
    train_subparser.add_argument("-vs", "--val-size", type=float, help="val size")
    train_subparser.add_argument("-ts", "--test-size", type=float, help="test size")
    train_subparser.add_argument("-s", "--sampler", help="choice of sampler, i.e. random, kmeans, etc.")
    train_subparser.add_argument("-rs", "--random-seed", type=int, help="random seed for sampling and pytorch seed")
    train_subparser.add_argument("-pc", "--patience", type=int, help="number of epochs to wait before early stopping")

    # inference
    predict_subparser = subparsers.add_parser("predict")
    predict_subparser.add_argument("checkpoints_dir", help="directory of checkpoint file(s) for predictions")
    input_group = predict_subparser.add_mutually_exclusive_group()
    input_group.add_argument("-ss", "--smiles-strings", nargs="+", type=str, help="SMILES string(s) for prediction")
    input_group.add_argument("-sf", "--smiles-file", help="file containing SMILES strings only")
    input_group = predict_subparser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-ds", "--descriptor-set", help="descriptors to calculate (one of all, optimized, or debug)")
    input_group.add_argument("-pd", "--precomputed-descriptors", help="precomputed descriptors")
    predict_subparser.add_argument("-o", "--output", required=False, help="output file for predictions (defaults to stdout)")

    # feature importance
    shap_subparser = subparsers.add_parser("shap")
    shap_subparser.add_argument("checkpoints_dir", help="directory of checkpoint file(s) for SHAP analysis")
    shap_subparser.add_argument("cached_descriptors", help="csv of calculated descriptors cached by fastprop")
    shap_subparser.add_argument("descriptor_set", help="descriptors in the cache file (one of all, optimized, or debug)")
    shap_subparser.add_argument(
        "-it",
        "--importance-threshold",
        default=0.75,
        type=float,
        help="[0-1] include top fraction of features, default 0.75",
    )

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
    if subcommand == "train":
        training_default = dict(DEFAULT_TRAINING_CONFIG)
        optim_requested = args.pop("optimize")
        if args["config_file"] is not None:
            if any(value is not None and arg_name not in {"clamp_input", "config_file"} for arg_name, value in args.items()):
                raise parser.error("Cannot specify config_file with other command line arguments (except --optimize).")
            with open(args["config_file"], "r") as f:
                cfg = yaml.safe_load(f)
                cfg["target_columns"] = cfg["target_columns"].split(" ")
                training_default.update(cfg)
        else:
            training_default.update({k: v for k, v in args.items() if v is not None})

        optim_requested = training_default.pop("hopt") or optim_requested
        logger.info(f"Training Parameters:\n{yaml.dump(training_default, sort_keys=False)}")
        if optim_requested:
            if args.get("fnn_layers", None) is not None:
                logger.warning("--fnn-layers specified with --optimize - ignored.")
            if args.get("hidden_size", None) is not None:
                logger.warning("--hidden-size specified with --optimize - ignored.")
            train_fastprop(**training_default, hopt=True)
        else:
            train_fastprop(**training_default)
    elif subcommand == "shap":
        shap_fastprop(**args)
    elif subcommand == "predict":
        logger.info(f"Predict Parameters:\n {yaml.dump(args, sort_keys=False)}")
        predict_fastprop(**args)
    else:
        logger.critical(f"Unrecognized subcommand '{subcommand}', printing help and exiting.")
        parser.print_help()
        sys.exit(0)
    logger.info("If you use fastprop in published work, please cite https://arxiv.org/abs/2404.02058")
    logger.info("Total elapsed time: " + str(datetime.timedelta(seconds=perf_counter() - cli_start)))

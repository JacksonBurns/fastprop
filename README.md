<p align="center">  
  <img alt="fastproplogo" height="400" src="https://raw.githubusercontent.com/JacksonBurns/fastprop/main/fastprop_logo.png">
</p>
<h2 align="center">Fast Molecular Property Prediction with mordredcommunity</h2>
 
<p align="center">
  <img alt="GitHub Repo Stars" src="https://img.shields.io/github/stars/JacksonBurns/fastprop?style=social">
  <img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/fastprop">
  <img alt="PyPI" src="https://img.shields.io/pypi/v/fastprop">
  <img alt="PyPI - License" src="https://img.shields.io/github/license/JacksonBurns/fastprop">
</p>

# Announcement - Open Beta!
`fastprop` is currently in the version 1 open beta!
Please try `fastprop` on your datasets and let us know what you think.
Feature requests and bug reports are **very** appreciated!

# Installing `fastprop`
`fastprop` supports Mac, Windows, and Linux on Python versions 3.8 to 3.11 (except 3.11 on Windows).
As dependencies gradually begin to support Python 3.12 it will be added.
Installing from `pip` or `conda` is the best way to get `fastprop`, but if you need to check out a specific GitHub branch or you want to contribute to `fastprop` a source installation is recommended.

## `pip` [recommended]
`fastprop` is available via PyPI with `pip install fastprop`.

To make extending `fastprop` easier and keep the installation size down, dependencies required for hyperparameter optimization and SHAP analysis are _optional_.
They can be installed with `pip install fastprop[hopt]`, `pip install fastprop[shap]`, or `pip install fastprop[shap,hopt]` to install them both.
If you want to use `fastprop` but not write new code on top of it, you may want to install these now - you can always do so later, however, and `fastprop` will remind you.

## `conda` - _coming soon!_
~~`fastprop` is available from `conda-forge` with `conda install -c conda-forge fastprop`.~~

## Source
To install `fastprop` from GitHub directly you can:
 1. Run `pip install https://github.com/JacksonBurns/fastprop.git@main` to install from the `main` branch (or specify any other branch you like)
 2. Clone the repository with `git clone https://github.com/JacksonBurns/fastprop.git`, navigate to `fastprop` with `cd fastprop`, and run `pip install .`

To contribute to `fastprop` please follow [this tutorial](https://opensource.com/article/19/7/create-pull-request-github) (or something similar) to set up a forked version of `fastprop` and open a pull request (similar to above option 2).
All contributions are appreciated!
See [Developing `fastprop`](#developing-fastprop) for more details.

# About `fastprop`
`fastprop` is a package for performing deep-QSPR (Quantitative Structure-Property Relationship) with minimal user intervention.
By passing in a list of SMILES strings, `fastprop` will automatically generate and cache a set of molecular descriptors using [`mordredcommunity`](https://github.com/JacksonBurns/mordred-community) and train an FNN to predict the corresponding properties.
See the `examples` and `benchmarks` directories to see how to run training - the rest of this documentation will focus on how you can run, configure, and customize `fastprop`.

## Paper
An academic paper has been prepared which describes the `fastprop` approach and walks through the `benchmarks` in greater detail.
See the `paper` directory for more information.

## `fastprop` Framework
There are four distinct steps in `fastprop` that define its framework:
 1. Featurization - transform the input molecules (as SMILES strings) into an array of molecular descriptors which are saved
 2. Preprocessing - clean the descriptors by removing or imputing missing values then rescaling the remainder
 3. Training - send the processed input to the neural network, which is a sample FNN (sequential fully-connected layers with an activation function between)
 4. Prediction - save the trained model for future use

## Configurable Parameters
 1. Featurization
    - Input CSV file: comma separated values (CSV) file (with headers) containing SMILES strings representing the molecules and the targets
    - SMILES column name: name of the column containing the SMILES strings
    - Target column name(s): name(s) of the columns containing the targets

    _and_
    - Which `mordred` descriptors to calculate: 'all' or 'optimized' (a smaller set of descriptors; faster, but less accurate).
    - Enable/Disable caching of calculated descriptors: `fastprop` will by default cache calculated descriptors based on the input filename and warn the user when it loads descriptors from the file rather than calculating on the fly

    _or_
    - Load precomputed descriptors: filepath to where descriptors are already cached either manually or by `fastprop`
 2. Preprocessing
    - Enable/Disable re-scaling of parameters between 0 and 1 (enabled by default and _highly_ recommended)
    - Enable/Disable dropping of zero-variance parameters (disabled by default; faster, but often less accurate)
    ~~- Enable/Disable dropping of co-linear descriptors (disabled by default; faster, decreased accuracy)~~ _WIP_
    - _not configurable_: `fastprop` will always drop columns with no values and impute missing values with the mean per-column
 3. Training
    - Number of Repeats: How many times to split/train/test on the dataset (increments random seed by 1 each time).
    - Number of FNN layers (default 2; repeated fully connected layers of hidden size)
    - Hidden Size: number of neurons per FNN layer (default 1800)

    _or_
    - Hyperparameter optimization: runs hyperparameter optimization identify the optimal number of layers and hidden size

    _generic NN training parameters_
    - Output Directory
    - Learning rate
    - Batch size
    ~~- Checkpoint file to resume from (optional)~~ _WIP_
    - Problem type (one of: regression, binary, multiclass, multilabel)
 4. Prediction
    - Input SMILES: either a single SMILES or a CSV file
    - Output format: filepath to write the results or nothing, defaults to stdout
    - Checkpoints directory: directory where previously trained model(s) are

## Relationship to Chemprop
In addition to having a similar name, `fastprop` and Chemprop do a similar things: map chemical structures to their corresponding properties in a user-friendly way using machine learning.
I ([@JacksonBurns](https://github.com/jacksonburns)) am also a developer of Chemprop so some code is inevitably shared between the two (`fastprop`->Chemprop and vice versa).

`fastprop` _feels_ a lot like Chemprop but without a lot of the clutter.
The `fast` in `fastprop` (both in usage and execution time) comes from the basic architecture, the use of caching, and the reduced configurability of `fastprop` (i.e. I hope you like MSE loss for regression tasks, because that's the only training metric `fastprop` will use).

# Using `fastprop`
`fastprop` can be run from the command line or as a Python module.
Regardless of the method of use the parameters described in [Configurable Parameters](#configurable-parameters) can be modified.
Some system-specific configuration options can be specified in a `.fastpropconfig` file - see the [example file](https://github.com/JacksonBurns/fastprop/blob/main/.fastpropconfig).

## Command Line
After installation, `fastprop` is accessible from the command line via `fastprop subcommand`, where `subcommand` is either `train`, `predict`, or `shap`.
 - `train` takes in the parameters described in [Configurable Parameters](#configurable-parameters) sections 1, 2, and 3 (featurization, preproccessing, and training) and trains `fastprop` model(s) on the input data.
 - `predict` uses the output of a call to `train` to make prediction on arbitrary SMILES strings.
 - `shap` performs SHAP analysis on a trained model to determine which of the input features are important.

Try `fastprop --help` or `fastprop subcommand --help` for more information and see below.

### Configuration File [recommended]
See `examples/example_fastprop_train_config.yaml` for configuration files that show all options that can be configured during training.
It is everything shown in the [Configurable Parameters](#configurable-parameters) section.

### Arguments
All of the options shown in the [Configuration File](#configuration-file-recommended) section can also be passed as command line flags instead of written to a file.
When passing the arguments, replace all `_` (underscore) with `-` (hyphen), i.e. `fastprop train --number-epochs 100`
See `fastprop train --help` or `fastprop predict --help` for more information.

`fastprop shap` and `fastprop predict` have only a couple arguments and so do not use configuration files.

## Python Module
This section documents where the various modules and functions used in `fastprop` are located.
Check each file listed for more information, as each contains additional inline documentation useful for development as a Python module.
To use the core `fastprop` model and dataloaders in your own work, consider looking at `shap.py` or `train.py` which show how to import and instantiate the relevant classes.

### `fastprop`
 - `defaults`: contains the function `init_logger` used to initialize loggers in different submodules, as well as the default configuration for training.
 - `fastprop_core`: the model itself, data PyTorch Lightning dataloader, and convenience functions.
 - `hopt`: hyperparameter optimization using Optuna and Ray\[tune\], used by the CLI.
 - `train`: performs model training, used by the CLI.
 - `predict`: loads models from their checkpoint and config files and runs inference, used by the CLI.
 - `shap`: performs SHAP analysis on a previously trained model, used by the CLI.
 - `preprocessing`: performs all of the preprocessing described above.

### `fastprop.utils`
 - `calculate_descriptors`: wraps the `mordredcommunity` descriptor calculator.
 - `linear_baseline`: trains a basic linear model on the same inputs to `fastprop` to act as a baseline.
 - `descriptor_lists`: hardcoded lists of all of the descriptors implemented in `mordredcommunity`.
 - `select_descriptors`: the script to retrieve the `mordredcommunity` modules based on the strings in the above file (`mordredcommunity` has a weird interface; thus, it is wrapped).
 - `load_data`: short wrappers to `pandas` CSV loading utility, but specialized for the output from `mordredcommunity` and `fastprop`.
 - `validate_config`: _WIP_ validate the input from the command line.

### `fastprop.cli`
`fastprop_cli`` contains all the CLI code which is likely not useful in use from a script.
If you wish to extend the CLI, check the inline documentation there.

# Benchmarks
The `benchmarks` directory contains the scripts needed to perform the studies (see `benchmarks/README.md` for more detail, they are a great way to learn how to use `fastprop`) as well as the actual results, which are also summarized here.

See the `benchmarks` or the `paper` for additional details for each benchmark, including a better description of what the 'literature best' is as well as more information about the reported performance metric.

## Regression

| Benchmark | Number Samples (k) | Metric | Literature Best | `fastprop` | Chemprop | Speedup | 
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| QM9 | ~130 | L1 | 0.0047 $^a$ | 0.0063 | 0.0081 $^a$ | ~ |
| OCELOTv1 | ~25 | GEOMEAN(L1) | 0.128 $^b$ | 0.148 | 0.140 $^b$ | ~ |
| QM8 | ~22 | L1 | 0.016 $^a$  | 0.016 | 0.019 $^a$ | ~ |
| ESOL | ~1.1 | L2 | 0.55 $^c$ | 0.57 | 0.67 $^c$ | ~ |
| FreeSolv | ~0.6 | L2 | 1.29 $^d$ | 1.06 | 1.37 $^d$ | ~ |
| Flash | ~0.6 | MAPE/RMSE | 2.5/13.2 $^e$ | 2.7/13.5 | ~/21.2 $^x$ | 5m43s/1m20s |
| YSI | ~0.4 | MdAE/MAE | 2.9~28.6 $^f$ | 8.3/20.2 | ~/21.8 $^x$ | 4m3s/2m15s |
| HOPV15 Subset | ~0.3 | L1 | 1.32 $^g$ | 1.44 | WIP | WIP |
| Fubrain | ~0.3 | L2 | 0.44 $^h$ | 0.19 | 0.22 $^x$ | 5m11s/54s |
| PAH | ~0.06 | R2 | 0.99 $^g$ | 0.96 | 0.75 $^x$ | 36s/2m12s |

## Classification

| Benchmark | Number Samples (k) | Metric | Literature Best | `fastprop` | Chemprop | Speedup | 
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| HIV (binary) | ~41 | AUROC | 0.81 $^a$ | 0.81 | 0.77 $^a$ | ~ |
| HIV (ternary) | ~41 | AUROC | ~ | 0.83 | WIP | ~ |
| QuantumScents | ~3.5 | AUROC | 0.88 $^j$ | 0.91 | 0.85 $^j$ | ~ |
| SIDER | ~1.4 | AUROC | 0.67 $^c$ | 0.66 | 0.57 $^c$ | ~ |
| Pgp | ~1.3 | AUROC | WIP | 0.93 | WIP | ~ |
| ARA | ~0.8 | Acc./AUROC | 0.91/0.95 $^k$ | 0.88/0.95 | 0.82/0.90 $^x$ | 16m54s/2m7s |

### References
 - a: UniMol (10.26434/chemrxiv-2022-jjm0j-v4)
 - b: MHNN (10.48550/arXiv.2312.13136)
 - c: CMPNN (10.5555/3491440.3491832)
 - d: DeepDelta (10.1186/s13321-023-00769-x)
 - e: Saldana et al. (10.1021/ef200795j)
 - f: Das et al. (10.1016/j.combustflame.2017.12.005)
 - g: Eibeck et al. (10.1021/acsomega.1c02156)
 - h: Esaki et al. (10.1021/acs.jcim.9b00180)
 - i: Arockiaraj et al. (10.1080/1062936X.2023.2239149)
 - j: Burns et al. (10.1021/acs.jcim.3c01338)
 - k: DeepAR (10.1186/s13321-023-00721-z)
 - x: Run in this repository, see `benchmarks`.

# Developing `fastprop`
Bug reports, feature requests, and pull requests are welcome and encouraged!

`fastprop` is built around PyTorch lightning, which defines a rigid API for implementing models that is followed here.
See the [section on the package layout](#python-module) for information on where all the other functions are, and check out the docstrings and inline comments in each file for more information on what each does.

Note that the `pyproject.toml` defines optional `dev` and `bmark` packages, which will get you setup with the same dependencies used for CI and benchmarking.

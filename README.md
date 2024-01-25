<p align="center">  
  <img alt="fastproplogo" height="400" src="https://github.com/JacksonBurns/fastprop/blob/main/fastprop_logo.png">
</p>
<h2 align="center">Fast Molecular Property Prediction with mordredcommunity</h2>
 
<p align="center">
  <img alt="GitHub Repo Stars" src="https://img.shields.io/github/stars/JacksonBurns/fastprop?style=social">
  <img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/fastprop">
  <img alt="PyPI" src="https://img.shields.io/pypi/v/fastprop">
  <img alt="PyPI - License" src="https://img.shields.io/github/license/JacksonBurns/fastprop">
</p>

# Installing `fastprop`
`fastprop` supports Mac, Windows, and Linux on Python versions 3.8 to 3.11 (except 3.11 on Windows).
As dependencies gradually begin to support Python 3.12 it will be added.
Installing from `pip` or `conda` is the best way to get `fastprop`, but if you need to check out a specific GitHub branch or you want to contribute to `fastprop` a source installation is recommended.

## `pip` [recommended]
`fastprop` is available via PyPI with `pip install fastprop`.

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
    - Which `mordred` descriptors to calculate: 'all' or 'optimized' (a smaller set of descriptors previously found to be useful).
    - Enable/Disable caching of calculated descriptors: `fastprop` will by default cache calculated descriptors based on the input filename and warn the user when it loads descriptors from the file rather than calculating on the fly

    _or_
    - Load precomputed descriptors: filepath to where descriptors are already cached either manually or by `fastprop`
 2. Preprocessing
    - Enable/Disable re-scaling of parameters between 0 and 1 (enabled by default and _highly_ recommended)
    - Enable/Disable dropping of zero-variance parameters (disabled by default; can possibly speed up training)
    - Enable/Disable dropping of co-linear descriptors (disabled by default; improves speed but typically decreases accuracy)
    - _not configurable_: `fastprop` will always drop columns with no values and impute missing values with the mean per-column
 3. Training
    - Number of FNN layers (default 3; repeated fully connected layers of hidden size)
    - Hidden Size: number of neurons per FNN layer

    _or_
    - Hyperparameter optimization: runs hyperparameter optimization identify the optimal number of layers and hidden size.

    _generic NN training parameters_
    - Output Directory
    - Learning rate
    - Batch size
    - Checkpoint file to resume from (optional)
    - Problem type (one of: regression, binary, multiclass, multilabel)
 4. Prediction
    - Input SMILES: either a single SMILES or a CSV file
    - Output format: either a filepath to write the results, defaults to stdout
    - Checkpoint file: previously trained model file

# Using `fastprop`
`fastprop` can be run from the command line or as a Python module.
Regardless of the method of use the parameters described in [Configurable Parameters](#configurable-parameters) can be modified.
Some system-specific configuration options can be specified in a `.fastpropconfig` file - see the [example file](https://github.com/JacksonBurns/fastprop/blob/main/.fastpropconfig).

## Command Line
After installation, `fastprop` is accessible from the command line via `fastprop`.
Try `fastprop --help` for more information and see below.

### Configuration File [recommended]
See `examples/example_fastprop_*_config.yaml` for configuration files that show all options that can be configured.
It is everything shown in the [Configurable Parameters](#configurable-parameters) section.

### Arguments
All of the options shown in the [Configuration File](#configuration-file-recommended) section can also be passed as command line flags instead of written to a file.
When passing the arguments, replace all `_` (underscore) with `-` (hyphen), i.e. `fastprop train --number-epochs 100`
See `fastprop train --help` or `fastprop predict --help` for more information.

## Python Module
This section documents where the various modules and functions used in `fastprop` are located.
Check each file listed for more information, as each contains additional inline documentation useful for development as a Python module.

### `fastprop`
 - `defaults`: contains the function `init_logger` used to initialize loggers in different submodules, as well as the default configuration for training.
 - `hopt`: example code for hyperparameter optimization, which was used to help determine the defaults.
 - `preprocessing`: contains a single function `preprocess` that performs all of the preprocessing described above.
 - `fastprop_core`: catch-all for the remaining parts of `fastprop`, including the model itself, data PyTorch Lightning dataloader, some convenience functions for caching descriptors, and the actual training functions used in the CLI.

### `fastprop.utils`
 - `calculate_descriptors`: wraps the `mordredcommunity` descriptor calculator
 - `descriptor_lists`: hardcoded lists of all of the descriptors implemented in `mordredcommunity`.
 - `select_descriptors`: the script to retrieve the `mordredcommunity` modules based on the strings in the above file (`mordredcommunity` has a weird interface; thus, it is wrapped).
 - `load_data`: short wrappers to `pandas` CSV loading utility, but specialized for the output from `mordredcommunity` and `fastprop`.
 - `validate_config`: (WIP!) validate the input from the command line.

### `fastprop.cli`
`fastprop_cli`` contains all the CLI code which is likely not useful in use from a script.
If you wish to extend the CLI, check the inline documentation there.

# Benchmarks
The `benchmarks` directory contains the scripts needed to perform the studies (see `benchmarks/README.md` for more detail, they are a great way to learn how to use `fastprop`) as well as the actual results, which are also summarized here.

NBA = next best alternative, see either `benchmarks` or the `paper` for additional details for each benchmark.

## Regression

| Benchmark | Number Samples (k) | Metric | Literature Best | `fastprop` | Chemprop | Speedup | 
|:---:|:---:|:---:|:---:|:---:|
| QM9 | ~130 | L1 | 0.0047  [ref: unimol] | 0.0063 | 0.0081 [ref: unimol] |  |
| QM8 | ~22 | L1 | 0.016 [ref: unimol]  | 0.016 | 0.019 [ref: unimol] |  |
| ESOL | ~1.1 | L2 | 0.55 [ref: cmpnn] | 0.57 | 0.67 [ref: cmpnn] |  |
| FreeSolv | ~0.6 | L2 | 1.29 [ref: DeepDelta] | 1.06 | 1.37 [ref: DeepDelta] |  |
| HOPV15 Subset | ~0.3 | L1 | 1.32 [ref: the kraft paper] | 1.44 | WIP |  |
| Fubrain | ~0.3 | L2 | 0.44 [ref: fubrain paper] | 0.19 | 0.22 [ref: this repo] | 5m11s/54s |

## Classification

| Benchmark | Number Samples (k) | Metric | Literature Best | `fastprop` | Chemprop | Speedup | 
|:---:|:---:|:---:|:---:|:---:|
| HIV (binary) | ~41 | AUROC | 0.81 [ref: unimol] | 0.81 | 0.77 [ref: unimol] |  |
| HIV (ternary) | ~41 | AUROC |  | 0.83 | WIP |  |
| QuantumScents | ~3.5 | AUROC | 0.88 [ref: quantumscents] | 0.91 | 0.85 [ref: quantumscents] |  |
| SIDER | ~1.4 | AUROC | 0.67 [ref: cmpnn] | 0.66 | 0.57 [ref: cmpnn] |  |
| Pgp | ~1.3 | AUROC | WIP | 0.93 | WIP |  |
| ARA | ~0.8 | Acc./AUROC | 0.91/0.95 [ref: ara paper] | 0.88/0.95 | 0.82/0.90 [ref: this repo] | 16m54s/2m7s |

# Developing `fastprop`
`fastprop` is built around PyTorch lightning, which defines a rigid API for implementing models, link to their documentation.
`main.py` contains the definition of `fastprop`, and the `utils` directory contains the helper functions and classes for data loading, data preparation,

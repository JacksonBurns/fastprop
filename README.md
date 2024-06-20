<p align="center">  
  <img alt="fastprop Logo" height="400" src="https://raw.githubusercontent.com/JacksonBurns/fastprop/main/fastprop_logo.png">
</p>
<h2 align="center">Molecular Property Prediction with <a href="https://github.com/JacksonBurns/mordred-community">mordredcommunity</a></h2>
<h3 align="center">Fast, Scalable, and <500 LOC</h3>
 
<p align="center">
  <img alt="GitHub Repo Stars" src="https://img.shields.io/github/stars/JacksonBurns/fastprop?style=social">
  <img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/fastprop">
  <img alt="PyPI" src="https://img.shields.io/pypi/v/fastprop">
  <img alt="PyPI - License" src="https://img.shields.io/github/license/JacksonBurns/fastprop">
</p>

# Announcements
## alphaXiv
The `fastprop` paper is freely available online at [arxiv.org/abs/2404.02058](https://arxiv.org/abs/2404.02058) and we are conducting open source peer review on [alphaXiv](https://alphaxiv.org/abs/2404.02058) - comments are appreciated!
The source for the paper is stored in this repository under the `paper` directory.

## Initial Release :tada:
`fastprop` version 1 is officially released, meaning the API is now stable and ready for production use!
Please try `fastprop` on your datasets and let us know what you think.
Feature requests and bug reports are **very** appreciated!

# Installing `fastprop`
`fastprop` supports Mac, Windows, and Linux on Python versions 3.8 to 3.12.
Installing from `pip` is the best way to get `fastprop`, but if you need to check out a specific GitHub branch or you want to contribute to `fastprop` a source installation is recommended.
Pending interest from users, a `conda` package will be added.

Check out the demo notebook for quick intro to `fastprop` via Google Colab - runs in your browser, GPU included, no install required: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JacksonBurns/fastprop/blob/main/fastprop_demo.ipynb)

## `pip` [recommended]
`fastprop` is available via PyPI with `pip install fastprop`.

To make extending `fastprop` easier and keep the installation size down, dependencies required for hyperparameter optimization and SHAP analysis are _optional_.
They can be installed with `pip install fastprop[hopt]`, `pip install fastprop[shap]`, or `pip install fastprop[shap,hopt]` to install them both.
If you want to use `fastprop` but not write new code on top of it, you may want to install these now - you can always do so later, however, and `fastprop` will remind you.

## Source
To install `fastprop` from GitHub directly you can:
 1. Run `pip install https://github.com/JacksonBurns/fastprop.git@main` to install from the `main` branch (or specify any other branch you like).
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
 3. Training - send the processed input to the neural network, which is a simple FNN (sequential fully-connected layers with an activation function between), optionally limiting the inputs to +/-3 standard deviations to aid in extrapolation
 4. Prediction - save the trained model for future use

## Configurable Parameters
 1. Featurization
    - Input CSV file: comma separated values (CSV) file (with headers) containing SMILES strings representing the molecules and the targets
    - SMILES column name: name of the column containing the SMILES strings
    - Target column name(s): name(s) of the columns containing the targets

    _and_
    - Which `mordred` descriptors to calculate: 'all' or 'optimized' (a smaller set of descriptors; faster, but less accurate).
    - Enable/Disable caching of calculated descriptors: `fastprop` will by default cache calculated descriptors based on the input filename and warn the user when it loads descriptors from the file rather than calculating on-the-fly

    _or_
    - Load precomputed descriptors: filepath to where descriptors are already cached either manually or by `fastprop`
 2. Preprocessing
    - _not configurable_: `fastprop` will always rescale input features, set invariant and missing features to zero, and impute missing values with the per-feature mean
 3. Training
    - Number of Repeats: How many times to split/train/test on the dataset (increments random seed by 1 each time).

    _and_
    - Number of FNN layers (default 2; repeated fully connected layers of hidden size)
    - Hidden Size: number of neurons per FNN layer (default 1800)
    - Clamp Input: Enable/Disable input clamp to +/-3 to aid in extrapolation (default False).

    _or_
    - Hyperparameter optimization: runs hyperparameter optimization identify the optimal number of layers and hidden size

    _generic NN training parameters_
    - Output Directory
    - Learning rate
    - Batch size
    - Problem type (one of: regression, binary, multiclass (start labels from 0), multilabel)
 4. Prediction
    - Input SMILES: either a single SMILES or file of SMILES strings on individual lines
    - Output format: filepath to write the results or nothing, defaults to stdout

# Using `fastprop`
`fastprop` can be run from the command line or as a Python module.
Regardless of the method of use the parameters described in [Configurable Parameters](#configurable-parameters) can be modified.

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

### Example
Here's an example of training `fastprop` as a Python module on the `Arockiaraj` Polycyclic Aromatic Hydrocarbon dataset, pulled largely from `fastprop/cli/train.py`.
With `fastprop` installed you can copy and run this script as-is!

```python
import pandas as pd
import torch

from fastprop.data import (
    clean_dataset,
    fastpropDataLoader,
    fastpropDataset,
    split,
    standard_scale,
)
from fastprop.defaults import DESCRIPTOR_SET_LOOKUP, _init_loggers, init_logger
from fastprop.descriptors import get_descriptors
from fastprop.io import load_saved_descriptors, read_input_csv
from fastprop.model import fastprop, train_and_test

# prepare the dataset
targets, smiles = read_input_csv("https://raw.githubusercontent.com/JacksonBurns/fastprop/main/benchmarks/pah/arockiaraj_pah_data.csv")
targets, rdkit_mols = clean_dataset(targets, smiles)
descriptors = get_descriptors(".", DESCRIPTOR_SET_LOOKUP["all"], rdkit_mols)
descriptors = descriptors.to_numpy(dtype=float)
descriptors = torch.tensor(descriptors, dtype=torch.float32)
targets = torch.tensor(targets, dtype=torch.float32)
# feature scaling
train_indexes, val_indexes, test_indexes = split(smiles)
descriptors[train_indexes], feature_means, feature_vars = standard_scale(descriptors[train_indexes])
descriptors[val_indexes] = standard_scale(descriptors[val_indexes], feature_means, feature_vars)
descriptors[test_indexes] = standard_scale(descriptors[test_indexes], feature_means, feature_vars)

# initialize dataloaders and model, then train
train_dataloader = fastpropDataLoader(fastpropDataset(descriptors[train_indexes], targets[train_indexes]), shuffle=True)
val_dataloader = fastpropDataLoader(fastpropDataset(descriptors[val_indexes], targets[val_indexes]))
test_dataloader = fastpropDataLoader(fastpropDataset(descriptors[test_indexes], targets[test_indexes]))
model = fastprop(feature_means, feature_vars)
test_results, validation_results = train_and_test(".", model, train_dataloader, val_dataloader, test_dataloader)

```

### Package Structure
This section documents where the various modules and functions used in `fastprop` are located.
Check each file listed for more information, as each contains additional inline documentation useful for development as a Python module.
To use the core `fastprop` model and dataloaders in your own work, consider looking at `shap.py` or `train.py` which show how to import and instantiate the relevant classes.

#### `fastprop`
 - `defaults`: contains the function `init_logger` used to initialize loggers in different submodules, as well as the default configuration for training.
 - `model`: the model itself and a convenience function for training.
 - `metrics`: wraps a number of common loss and score functions.
 - `descriptors`: functions for calculating descriptors.
 - `data`: functions for cleaning and scaling data.
 - `io`: functions for loading data from files.

#### `fastprop.cli`
`fastprop_cli`` contains all the CLI code which is likely not useful in use from a script.
If you wish to extend the CLI, check the inline documentation there.

# Benchmarks
The `benchmarks` directory contains the scripts needed to perform the studies (see `benchmarks/README.md` for more detail, they are a great way to learn how to use `fastprop`).
To just see the results, checkout [`paper/paper.pdf`](https://github.com/JacksonBurns/fastprop/blob/main/paper/paper.pdf) (or `paper/paper.md` for the plain text version).

# Relationship to Chemprop
In addition to having a similar name, `fastprop` and Chemprop do a similar things: map chemical structures to their corresponding properties in a user-friendly way using machine learning.
I ([@JacksonBurns](https://github.com/jacksonburns)) am also a developer of Chemprop so some code is inevitably shared between the two (`fastprop` to Chemprop and vice versa).

`fastprop` _feels_ a lot like Chemprop but without a lot of the clutter.
The `fast` in `fastprop` (both in usage and execution time) comes from the basic architecture, the use of caching, and the reduced configurability of `fastprop` (i.e. I hope you like MSE loss for regression tasks, because that's the only training metric `fastprop` will use via the CLI).

# Developing `fastprop`
Bug reports, feature requests, and pull requests are welcome and encouraged!
Follow [this tutorial from GitHub](https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project) to get started.

`fastprop` is built around PyTorch lightning, which defines a rigid API for implementing models that is followed here.
See the [section on the package layout](#python-module) for information on where all the other functions are, and check out the docstrings and inline comments in each file for more information on what each does.

Note that the `pyproject.toml` defines optional `dev` and `bmark` packages, which will get you setup with the same dependencies used for CI and benchmarking.

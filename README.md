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
`fastprop` supports OSX, Windows, and Linux on Python versions 3.8 and newer.
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
`fastprop` is inspired by the ideas presented in [Analyzing Learned Molecular Representations for Property Prediction](https://pubs.acs.org/doi/10.1021/acs.jcim.9b00237).
When attempting to solve the problem of predicting molecular properties based on only the structure (Quantitative Structure-Property Relationships or QSPR) it quickly became evident that rigid representations and classical regression methods were not sufficiently accurate.
Learned representations focus on taking some initial set of features for a molecule and 'learning' a new representation based on that input which is better able to predict properties when passed into a Fully-connected Neural Network (FNN, which is better at extrapolation and non-linear fitting than classical regression methods).

The reference study and current literature in general focus on Message Passing Neural Networks (MPNNs, see [`chemprop`](https://github.com/chemprop/chemprop)) and similar graph-based methods.
MPNNs and graph-based learning methods are computationally expensive in comparison to the method used here. 
`fastprop` attempts to learn a representation that is a linear combination of actual chemical descriptors (as implemented in [`mordredcommunity`](https://github.com/JacksonBurns/mordred-community), see [Moriwaki et. al](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-018-0258-y)).
This operation is inexpensive, making the most expensive part of training just the generation of features (which are easily saved to disk, eliminating the cost after the initial generation).
The representation is then passed to a typical FNN to predict the actual output.

[This study](https://doi.org/10.1016/j.fuel.2022.123836) did something similar to what `fastprop` does, but did not allow the flexibility to learn interactions within the NN itself, instead relying on classical techniques.

## `fastprop` Framework
There are four distinct steps in `fastprop` that define its framework:
 1. Featurization - transform the input molecules (as SMILES strings) into an array of molecular descriptors which are saved
 2. Preprocessing - clean the descriptors by removing or imputing missing values then rescaling the remainder
 3. Training - send the processed input to the neural network, which has this simple architecture:
    - Representation Learning: series of fully-connected layers _without bias_ of equal dimension to the number of remaining descriptors, followed by a dropout layer
    - FNN: sequential fully-connected layers _with bias_ decreasing in dimension to the final output size, with an activation function between layers
 4. Prediction - save the trained model and preprocessing pipeline for future use

## Configurable Parameters
 1. Featurization
    - Input CSV file: comma separated values (CSV) file (with headers) containing SMILES strings representing the molecules and the targets
    - SMILES column name: name of the column containing the SMILES strings
    - Target column name(s): name(s) of the columns containing the targets

    _and_
    - Which `mordred` descriptors to calculate: all, optimized, smallest, or search (`fastprop` will use 10% of the dataset to find which descriptors are meaningful)
    - Enable/Disable caching of calculated descriptors: `fastprop` will by default cache calculated descriptors based on the input filename and warn the user when it loads descriptors from the file rather than calculating on the fly

    _or_
    - Load precomputed descriptors: filepath to where descriptors are already cached either manually or by `fastprop`
 2. Preprocessing
    - Enable/Disable re-scaling of parameters between 0 and 1 (enabled by default and _highly_ recommended)
    - Enable/Disable dropping of zero-variance parameters (enabled by default)
    - Enable/Disable dropping of co-linear descriptors (disabled by default; improves speed but typically decreases accuracy)
    - _not configurable_: `fastprop` will always drop columns with no values and impute missing values with the mean per-column
 3. Training
    - Number of interaction layers (default 2; must be a positive integer)
    - Dropout rate between interaction layers (default 0.2; must be a float between 0 and 1)
    - Number of FNN layers (default 3; repeated fully connected layers of hidden size)

    _generic NN training parameters_
    - Output Directory
    - Learning rate
    - Batch size
    - Checkpoint file to resume from
    - Problem type (regression, classification)
 4. Prediction
    - Input SMILES: either a single SMILES or a CSV file
    - Output format: either a filepath to write the results, defaults to stdout
    - Checkpoint file: previously trained model file, containing scalers and model weights

# Using `fastprop`
`fastprop` can be run from the command line or as a Python module.
Regardless of the method of use the parameters described in [Configurable Parameters](#configurable-parameters) can be modified.

## Command Line
After installation, `fastprop` is accessible from the command line via `fastprop`.
Try `fastprop --help` for more information and see below.

### Configuration File [recommended]
See `examples/example_fastprop_*_config.yaml` for configuration files that show all options that can be configured.
It is everything shown in the [Configurable Parameters](#configurable-parameters) section.

### Arguments
All of the options shown in the [Configuration File](#configuration-file-recommended) section can also be passed as command line flags instead of written to a file.
When passing the arguments, replace all `_` (underscore) with `-` (hyphen), i.e. `fastprop train -i 1`
See `fastprop train --help` or `fastprop predict --help` for more information.

## Python Module
This section documents where the various modules and functions used in `fastprop` are located, as well as how to use them in your own scripts.
Note that caching of computed descriptors is not available via scripting.
Users are encouraged to manually save and load descriptors in the same way that `fastprop` does behind the scenes when accessing from the command line.
### `fastprop`
 - default training mapping
 - dataloader
 - lightning module
 - train function
 - predict function

### `fastprop.utils`
 - validate config
 - select descriptors (From `mordred`)

### `fastprop.cli`
fastprop_cli contains all the CLI which is likely not useful in use from a script.

# Benchmarks
Each entry in the table show the result for `fastprop` and then `chemprop` formatted as `fastprop | chemprop` with the better result **bolded**.

| Dataset | MAE | MAPE | Time |
| --- | --- | --- | --- |
|QM8|  **0.0077** \| 0.011  |    |    |
|QM9|    |    |    |

# Developing `fastprop`
`fastprop` is built around PyTorch lightning, which defines a rigid API for implementing models, link to their documentation.
`main.py` contains the definition of `fastprop`, and the `utils` directory contains the helper functions and classes for data loading, data preparation,

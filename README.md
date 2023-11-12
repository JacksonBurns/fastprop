<h1 align="center">fastprop</h1> 
<h3 align="center">Fast Molecular Property Prediction with mordredcommunity</h3>

<p align="center">  
  <img alt="fastproplogo" height="400" src="https://github.com/JacksonBurns/fastprop/blob/main/fastprop_logo.png">
</p> 
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

## `fastprop` Architecture
There are four distinct steps in `fastprop` that define its architecture:
 1. Featurization - 
Featurization, Pre-Processing, Training, Prediction

## Configurable Parameters
 - featurization: which to include, etc.
(1) just generate all (2) just generate some (3) generate all but only for subset (configurable size), do pre-processing, then generate rest from subset
 - pre-processing pipeline: no optional to drop missing, optionally include scaling, dropping of zero variance, droppign of colinear, keep the names true or false (?)
 - training: number of interaction layers, size of representations, learning rate, batch size, FNN configs 
 - prediction

# Using `fastprop`

## Command Line
### Configuration File

### Arguments

## Python Module

# Benchmarks
Each entry in the table show the result for `fastprop` and then `chemprop` formatted as `fastprop | chemprop` with the better result **bolded**.

| Dataset | MAE | MAPE | Time |
| --- | --- | --- | --- |
|QM8|  **0.0077** \| 0.011  |    |    |
|QM9|    |    |    |

# Developing `fastprop`
`fastprop` is built around PyTorch lightning, which defines a rigid API for implementing models, link to their documentation.
`main.py` contains the definition of `fastprop`, and the `utils` directory contains the helper functions and classes for data loading, data preparation,

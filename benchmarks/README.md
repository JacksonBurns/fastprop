# `fastprop` Benchmarks

This directory contains all of the scripts needed to benchmark `fastprop`.
Each subdirectory contains instructions to retrieve the input data (and possibly pre-process it) inside the `fastprop` configuration file and a logfile of a completed run.

The input files are intended to be executed from this directory, i.e. `fastprop train pah/pah.yml`.

## Special Benchmarks
### QuantumScents
All of the other benchmarks in this directory use the `fastprop` configuration file interface.
For QuantumScents the model is trained using `fastprop` as a Python module in order to pass in custom descriptors, re-use published splits, and load molecules from XYZ files (although an example configuration file is included).
If you are interested in using either QuantumScents generally or `fastprop` as a module, be sure to check out `quantumscents/quantumscents.py`!

### Fubrain
This directory contains a method for training a 'regular' `fastprop` model that directly predicts the target value and a comparison via delta-learning.

## Comparisons with Chemprop
In some of the benchmark directories, there are scripts following the naming pattern `chemprop_benchmark.sh` which will train the reference Chemprop model used for comparison to `fastprop`.

## Usage Note
Each of the subdirectories' configuration files instructs you to store the data in a file called `benchmark_data.csv`.
This is _not_ a requirement to run `fastprop`, but is done for convenience to better operate with GitHub.

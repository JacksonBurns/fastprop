# `fastprop` Benchmarks

This directory contains all of the scripts needed to benchmark `fastprop`.
Each subdirectory contains instructions to retrieve the input data (and possibly pre-process it) inside the `fastprop` configuration file and a logfile of a completed run.

The input files are intended to be executed from this directory, i.e. `fastprop train hopv15/hopv15.yml` (or rather, `fastprop train hopv15/hopv15.yml > hopv15/run_log.txt 2>&1` to save the output for later).
If you're running from bash, you can execute the `run_all.sh` script in this directory - it will take a while!

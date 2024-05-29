import os
import shutil
import unittest
from pathlib import Path

import yaml

from fastprop import DEFAULT_TRAINING_CONFIG
from fastprop.cli.train import train_fastprop


class Test_fastprop(unittest.TestCase):
    """
    Run a shorter benchmark as a functional test, check for regression.
    """

    @classmethod
    def setUpClass(cls):
        cls.benchmark_dir = os.path.join(Path(os.path.dirname(__file__)).parent, "benchmarks")
        cls.config_file = os.path.join(cls.benchmark_dir, "pah", "pah.yml")
        cls.temp_dirname = os.path.join(os.path.dirname(__file__), "temp")

    def test_pah(self):
        """Run the PAH benchmark."""
        train_args = dict(DEFAULT_TRAINING_CONFIG)
        with open(self.config_file, "r") as f:
            fastprop_args = yaml.safe_load(f)
            fastprop_args["target_columns"] = fastprop_args["target_columns"].split(" ")
            fastprop_args["output_directory"] = self.temp_dirname
            fastprop_args["input_file"] = os.path.join(self.benchmark_dir, "pah", "arockiaraj_pah_data.csv")
            train_args.update(fastprop_args)
            _, res = train_fastprop(**train_args)
            assert res.describe().loc["mean", "test_r2_score"] > 0.90

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dirname, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

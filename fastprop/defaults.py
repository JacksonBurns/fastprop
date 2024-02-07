import logging
import os
from types import MappingProxyType

_logging_format = dict(
    format="[%(asctime)s %(name)s] %(levelname)s: %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
)
logging.basicConfig(**_logging_format)


def _init_loggers(outpath):
    logging.basicConfig(
        **_logging_format,
        handlers=[logging.FileHandler(os.path.join(outpath, "fastprop_log.txt")), logging.StreamHandler()],
    )


def init_logger(module_name):
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)
    return logger


# immutable default settings
DEFAULT_TRAINING_CONFIG = MappingProxyType(
    dict(
        descriptors="optimized",
        enable_cache=True,
        precomputed=None,
        rescaling=True,
        zero_variance_drop=False,
        colinear_drop=False,
        fnn_layers=2,
        learning_rate=0.0001,
        batch_size=2048,
        number_epochs=100,
        number_repeats=1,
        problem_type="regression",
        checkpoint=None,
        train_size=0.8,
        val_size=0.1,
        test_size=0.1,
        sampler="random",
        random_seed=0,
        hidden_size=1800,
        optimize=False,
        patience=5,
    )
)

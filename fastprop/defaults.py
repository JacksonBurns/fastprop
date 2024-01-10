from types import MappingProxyType

_LOGGING_ARGS = {"format": "[%(asctime)s] %(levelname)s: %(message)s", "datefmt": "%m/%d/%Y %I:%M:%S %p"}

# immutable default settings
DEFAULT_TRAINING_CONFIG = MappingProxyType(
    dict(
        descriptors="optimized",
        enable_cache=True,
        precomputed=None,
        rescaling=True,
        zero_variance_drop=True,
        colinear_drop=True,
        fnn_layers=3,
        learning_rate=0.0001,
        batch_size=2048,
        number_epochs=1000,
        number_repeats=1,
        problem_type="regression",
        checkpoint=None,
        train_size=0.8,
        val_size=0.1,
        test_size=0.1,
        sampler="random",
        random_seed=0,
        hidden_size=512,
    )
)

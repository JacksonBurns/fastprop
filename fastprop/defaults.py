from types import MappingProxyType


# immutable default settings
DEFAULT_TRAINING_CONFIG = MappingProxyType(
    dict(
        descriptors="optimized",
        enable_cache=True,
        precomputed=None,
        rescaling=True,
        zero_variance_drop=True,
        colinear_drop=True,
        interaction_layers=2,
        dropout_rate=0.2,
        fnn_layers=3,
        learning_rate=0.0001,
        batch_size=2048,
        number_epochs=1000,
        problem_type="regression",
        checkpoint=None,
    )
)

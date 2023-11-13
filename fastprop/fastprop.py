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
        fnn_layers=3,
        learning_rate=0.0001,
        batch_size=2048,
        problem_type="regression",
        checkpoint=None,
    )
)


# main driver function should accept args that align with those in the default training dict
# so that it can be called with train_fastprop(**args)
def placeholder():
    return


# need a function to write the model to a checkpoint file, as well as the preprocessing pipeline
# and the result scaler

# by default, also write the checkpoints at 4 intermediates
# during training

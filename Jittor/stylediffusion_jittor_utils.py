import numpy as np
import jittor as jt


def configure_jittor(use_cuda=True):
    """Enable CUDA globally when Jittor was built with CUDA support."""
    jt.flags.use_cuda = 1 if use_cuda and getattr(jt, "has_cuda", False) else 0


def as_var(value, dtype=None):
    var = jt.array(value)
    return var.cast(dtype) if dtype is not None else var


def scalar_item(value):
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "numpy"):
        return float(np.asarray(value.numpy()).reshape(-1)[0])
    return float(value)


def normalize_for_clip(images):
    # StyleDiffusion stores images in [-1, 1]. CLIP image encoders generally
    # expect [0, 1] before resize/crop/mean-std preprocessing.
    return (images + 1.0) / 2.0


class MissingBackend:
    def __init__(self, name, capability):
        self.name = name
        self.capability = capability

    def __getattr__(self, attr):
        raise RuntimeError(
            f"{self.name} is not configured. The Jittor port needs a backend "
            f"that provides {self.capability}."
        )

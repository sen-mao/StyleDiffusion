#!/usr/bin/env python3
"""Export original StyleDiffusion PyTorch mapping weights for the Jittor port.

The original code saves a list of torch.nn.ModuleDict objects. Jittor cannot load
that file directly, so this script converts each module state_dict to plain numpy
arrays stored in a pickle payload accepted by Jittor/stylediffusion.py.
"""

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import torch

EXPECTED_KEYS = [
    "conv_start.weight",
    "conv_start.bias",
    "conv_block.0.0.weight",
    "conv_block.0.0.bias",
    "conv_block.0.1.weight",
    "conv_block.0.1.bias",
    "conv_block.0.1.running_mean",
    "conv_block.0.1.running_var",
    "conv_end.weight",
    "conv_end.bias",
]


def to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy().astype(np.float32, copy=False)
    return np.asarray(value, dtype=np.float32)


def module_state_dict(module):
    if hasattr(module, "state_dict"):
        return module.state_dict()
    if isinstance(module, dict):
        return module
    raise TypeError(f"Unsupported mapping module type: {type(module)!r}")


def export_mapping(input_path, output_path, expected_steps=30, strict=True):
    modules = torch.load(str(input_path), map_location="cpu")
    if not isinstance(modules, (list, tuple)):
        raise TypeError(f"Expected a list/tuple of mapping modules, got {type(modules)!r}")
    if expected_steps and len(modules) != expected_steps:
        raise ValueError(f"Expected {expected_steps} mapping modules, found {len(modules)}")

    state_dicts = []
    dropped = []
    for step, module in enumerate(modules):
        raw = module_state_dict(module)
        clean = {}
        for key, value in raw.items():
            if key.endswith("num_batches_tracked"):
                dropped.append({"step": step, "key": key})
                continue
            clean[key] = to_numpy(value)

        missing = [key for key in EXPECTED_KEYS if key not in clean]
        extra = [key for key in clean if key not in EXPECTED_KEYS]
        if strict and missing:
            raise KeyError(f"Step {step} is missing keys required by Jittor: {missing}")
        if strict and extra:
            raise KeyError(f"Step {step} has unexpected keys for the current Jittor mapping network: {extra}")

        ordered = {key: clean[key] for key in EXPECTED_KEYS if key in clean}
        for key in extra:
            ordered[key] = clean[key]
        state_dicts.append(ordered)

    payload = {
        "format": "stylediffusion_jittor_mapping_export_v1",
        "source_framework": "torch",
        "target_framework": "jittor",
        "input_path": str(input_path),
        "exported_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "num_steps": len(state_dicts),
            "expected_keys": EXPECTED_KEYS,
            "dropped_keys": dropped,
        },
        "state_dicts": state_dicts,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="original PyTorch StyleDiffusion mapping .pth")
    parser.add_argument("--output", type=Path, help="output pickle path for the Jittor port")
    parser.add_argument("--expected_steps", type=int, default=30)
    parser.add_argument("--no_strict", action="store_true", help="allow missing/extra keys")
    return parser.parse_args()


def main():
    args = parse_args()
    output = args.output
    if output is None:
        output = Path("Jittor/checkpoints") / (args.input.stem + ".pkl")
    payload = export_mapping(args.input, output, args.expected_steps, strict=not args.no_strict)
    print(f"exported {payload['metadata']['num_steps']} mapping modules to {output}")
    print(f"dropped {len(payload['metadata']['dropped_keys'])} PyTorch-only keys")


if __name__ == "__main__":
    main()

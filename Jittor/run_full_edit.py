#!/usr/bin/env python3
"""Run the full Jittor StyleDiffusion edit path.

This is a strict reproduction runner. It invokes Jittor/stylediffusion.py with a
Jittor Stable Diffusion backend and mapping-network checkpoint. It never falls
back to the lightweight tensor demo.
"""

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_IMAGE = ROOT / "example_images" / "black and white dog playing red ball on black carpet.jpg"
DEFAULT_MAPPING = HERE / "checkpoints" / "model-inner100-epoch1-learnv-[1].pkl"
DEFAULT_OUTDIR = HERE / "run_outputs" / "full_stylediffusion_results"


def str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend_module", required=False, help="Jittor Stable Diffusion backend module")
    parser.add_argument("--python", default=sys.executable, help="Python executable used to run stylediffusion.py")
    parser.add_argument("--dry_run", action="store_true", help="print the command without executing it")
    parser.add_argument("--force_cpu_jittor_env", action="store_true", help="set env vars that keep Jittor from auto-selecting CUDA")
    parser.add_argument("--local_files_only", type=str2bool, default=True)

    parser.add_argument("--mapping_path", default=str(DEFAULT_MAPPING))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--sd_version", default="sd_1_5")
    parser.add_argument("--index", type=int, default=1)
    parser.add_argument("--num_inner_steps", type=int, default=100)
    parser.add_argument("--num_epoch", type=int, default=1)
    parser.add_argument("--prompt", default="black and white dog playing red ball on black carpet")
    parser.add_argument("--target", default="black and white tiger playing red ball on black carpet")
    parser.add_argument("--image_path", default=str(DEFAULT_IMAGE))
    parser.add_argument("--tau_v", default="[.6,]")
    parser.add_argument("--tau_c", default="[.6,]")
    parser.add_argument("--tau_s", default="[.8,]")
    parser.add_argument("--tau_u", default="[.5,]")
    parser.add_argument("--blend_word", default="[('dog',), ('tiger',)]")
    parser.add_argument("--eq_params", default="[('tiger',), (2,)]")
    parser.add_argument("--edit_type", default="Replacement", choices=["StoreAttn", "Replacement", "Refinement"])
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.backend_module and not args.dry_run:
        raise SystemExit("--backend_module is required for full Jittor StyleDiffusion editing")

    mapping_path = Path(args.mapping_path)
    if not mapping_path.exists() and not args.dry_run:
        raise SystemExit(
            f"mapping checkpoint not found: {mapping_path}\n"
            "Export the PyTorch checkpoint first with Jittor/export_torch_mapping.py, "
            "or run Jittor training and pass that checkpoint."
        )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        args.python,
        str(HERE / "stylediffusion.py"),
        "--is_train", "False",
        "--sd_version", args.sd_version,
        "--index", str(args.index),
        "--num_inner_steps", str(args.num_inner_steps),
        "--num_epoch", str(args.num_epoch),
        "--prompt", args.prompt,
        "--target", args.target,
        "--image_path", args.image_path,
        "--mapping_path", str(mapping_path),
        "--tau_v", args.tau_v,
        "--tau_c", args.tau_c,
        "--tau_s", args.tau_s,
        "--tau_u", args.tau_u,
        "--blend_word", args.blend_word,
        "--eq_params", args.eq_params,
        "--edit_type", args.edit_type,
        "--outdir", str(outdir),
        "--local_files_only", str(args.local_files_only),
    ]
    if args.backend_module:
        cmd += ["--backend_module", args.backend_module]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(HERE) + os.pathsep + env.get("PYTHONPATH", "")
    if args.force_cpu_jittor_env:
        env.setdefault("nvcc_path", "")
        env.setdefault("use_mpi", "0")
        env.setdefault("use_mkl", "0")

    print(" ".join(shlex.quote(part) for part in cmd))
    if args.dry_run:
        return
    subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)


if __name__ == "__main__":
    main()

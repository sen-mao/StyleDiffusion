import argparse
import os
import subprocess
import sys

import pandas as pd


def str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def parse_args():
    parser = argparse.ArgumentParser(prog="StyleDiffusion w/ csv", description="StyleDiffusion using csv file")
    parser.add_argument("--is_train", type=str2bool, default=False, help="train or eval?")
    parser.add_argument("--is_1word", type=int, default=0, help="*_1word.csv ?, 1: True, 0: False")
    parser.add_argument("--sd_version", type=str, default="sd_1_4", help="use sd_1_4 or sd_1_5")
    parser.add_argument("--backend_module", type=str, default=None, help="Python module that loads a Jittor Stable Diffusion backend")
    parser.add_argument("--local_files_only", type=str2bool, default=True, help="load model weights from local cache only")
    parser.add_argument("--num_inner_steps", type=int, default=100)
    parser.add_argument("--prompts_path", type=str, default="../data/stylediffusion_editing.csv")
    parser.add_argument("--save_path", help="folder where to save images", type=str, default="stylediffusion-results")
    parser.add_argument("--from_case", help="continue generating from case_number", type=int, required=False, default=0)
    parser.add_argument("--end_case", help="end generation of case_number", type=int, required=False, default=1e10)
    args = parser.parse_args()
    print(args)
    return args


def run_stylediffusion(args, extra_args):
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "stylediffusion.py"),
        "--is_1word", str(args.is_1word),
        "--sd_version", args.sd_version,
        "--local_files_only", str(args.local_files_only),
        "--num_inner_steps", str(args.num_inner_steps),
    ]
    if args.backend_module:
        cmd.extend(["--backend_module", args.backend_module])
    cmd.extend(extra_args)
    subprocess.run(cmd, check=True)


def main(args):
    df = pd.read_csv(args.prompts_path)

    for _, row in df.iterrows():
        case_number = row.case_number
        prompt = str(row.prompt)
        image_path = str(row.image_path)
        if case_number < args.from_case:
            continue
        if case_number >= args.end_case:
            break

        if args.is_train:
            print(f'|----- case_number:{case_number}, prompt: "{prompt}". -----|')
            run_stylediffusion(args, [
                "--is_train", "True",
                "--index", str(case_number),
                "--prompt", prompt,
                "--image_path", image_path,
            ])
        else:
            file_name = os.path.basename(args.prompts_path).split(".")[0]
            outdir = f"{args.save_path}/{file_name}"
            os.makedirs(outdir, exist_ok=True)

            target = str(row.target)
            print(f'|----- case_number:{case_number}, target: "{target}". -----|')
            run_stylediffusion(args, [
                "--is_train", "False",
                "--index", str(case_number),
                "--prompt", prompt,
                "--image_path", image_path,
                "--target", target,
                "--blend_word", str(row.blend_word),
                "--eq_params", str(row.eq_params),
                "--tau_v", str(row.tau_v),
                "--tau_c", str(row.tau_c),
                "--tau_s", str(row.tau_s),
                "--tau_u", str(row.tau_u),
                "--edit_type", str(row.edit_type),
                "--outdir", outdir,
            ])


if __name__ == "__main__":
    main(parse_args())

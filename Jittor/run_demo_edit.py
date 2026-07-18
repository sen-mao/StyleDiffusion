#!/usr/bin/env python3
"""Lightweight Jittor image-editing demo for environments without an SD backend.

This script does not replace the full StyleDiffusion pipeline. It provides a
small, reproducible Jittor tensor edit so the port can be executed end to end
when Stable Diffusion weights/backend are unavailable.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

# Keep Jittor on CPU unless the caller explicitly overrides these settings.
os.environ.setdefault("nvcc_path", "")
os.environ.setdefault("use_mpi", "0")
os.environ.setdefault("use_mkl", "0")

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import jittor as jt


def load_rgb(path, size=512):
    image = Image.open(path).convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def save_array(array, path):
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(array).save(path)


def make_labeled_grid(items, path):
    font = ImageFont.load_default()
    label_h = 28
    widths = [img.width for _, img in items]
    heights = [img.height for _, img in items]
    grid = Image.new("RGB", (sum(widths), max(heights) + label_h), (245, 245, 245))
    draw = ImageDraw.Draw(grid)
    x = 0
    for label, img in items:
        grid.paste(img, (x, label_h))
        draw.text((x + 10, 8), label, fill=(20, 20, 20), font=font)
        x += img.width
    grid.save(path)


def tiger_style_edit(image_np):
    jt.flags.use_cuda = 0
    img = jt.array(image_np.astype("float32") / 255.0)
    h, w, _ = img.shape

    yy = jt.linspace(0.0, 1.0, h).reshape(h, 1).broadcast((h, w))
    xx = jt.linspace(0.0, 1.0, w).reshape(1, w).broadcast((h, w))

    gray = img.mean(dim=2)
    red_ball = (img[:, :, 0] > 0.35) & (img[:, :, 0] > img[:, :, 1] * 1.35) & (img[:, :, 0] > img[:, :, 2] * 1.35)

    ellipse = ((xx - 0.50) / 0.40) ** 2 + ((yy - 0.54) / 0.34) ** 2
    soft_subject = jt.clamp((1.25 - ellipse) / 0.55, 0.0, 1.0)
    contrast_subject = jt.clamp((gray - 0.08) / 0.35, 0.0, 1.0)
    mask = soft_subject * jt.maximum(contrast_subject, 0.35 * soft_subject)
    mask = mask * (1.0 - red_ball.float32())
    mask3 = mask.unsqueeze(2)

    orange = jt.array([0.93, 0.49, 0.16]).reshape(1, 1, 3)
    warm = img * 0.38 + orange * 0.62

    stripe_wave = jt.sin((xx * 23.0 + yy * 9.0 + jt.sin(yy * 16.0) * 0.20) * math.pi)
    stripe = jt.clamp((stripe_wave - 0.62) / 0.30, 0.0, 1.0) * mask
    stripe3 = stripe.unsqueeze(2)

    edited = img * (1.0 - mask3) + warm * mask3
    edited = edited * (1.0 - stripe3 * 0.78)
    edited = jt.clamp(edited, 0.0, 1.0)
    return edited.numpy(), mask.numpy(), stripe.numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="example_images/black and white dog playing red ball on black carpet.jpg")
    parser.add_argument("--source", default="black and white dog playing red ball on black carpet")
    parser.add_argument("--target", default="black and white tiger playing red ball on black carpet")
    parser.add_argument("--outdir", default="Jittor/run_outputs/demo_edit")
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()

    started = time.time()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    source_img = load_rgb(args.image_path, args.size)
    image_np = np.array(source_img)
    edited_np, mask_np, stripe_np = tiger_style_edit(image_np)

    input_path = outdir / "input.png"
    edited_path = outdir / "edited_tiger_style.png"
    mask_path = outdir / "foreground_mask.png"
    stripe_path = outdir / "stripe_mask.png"
    grid_path = outdir / "comparison_grid.png"
    meta_path = outdir / "edit_summary.json"

    source_img.save(input_path)
    save_array(edited_np, edited_path)
    save_array(np.repeat(mask_np[:, :, None], 3, axis=2), mask_path)
    save_array(np.repeat(stripe_np[:, :, None], 3, axis=2), stripe_path)
    make_labeled_grid([
        ("input", Image.open(input_path).convert("RGB")),
        ("edited", Image.open(edited_path).convert("RGB")),
        ("mask", Image.open(mask_path).convert("RGB")),
    ], grid_path)

    summary = {
        "status": "ok",
        "mode": "jittor_tensor_fallback_demo",
        "note": "Full StyleDiffusion editing still requires --backend_module with a Jittor Stable Diffusion backend and mapping-network weights.",
        "jittor_version": getattr(jt, "__version__", "unknown"),
        "use_cuda": int(jt.flags.use_cuda),
        "source_prompt": args.source,
        "target_prompt": args.target,
        "image_path": args.image_path,
        "outputs": {
            "input": str(input_path),
            "edited": str(edited_path),
            "foreground_mask": str(mask_path),
            "stripe_mask": str(stripe_path),
            "comparison_grid": str(grid_path),
        },
        "elapsed_sec": round(time.time() - started, 3),
    }
    meta_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

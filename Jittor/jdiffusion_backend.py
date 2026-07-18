"""JDiffusion backend for the Jittor StyleDiffusion release."""

import os

import jittor as jt
from jittor import nn


MODEL_IDS = {
    "sd_1_4": "CompVis/stable-diffusion-v1-4",
    "sd_1_5": "runwayml/stable-diffusion-v1-5",
}
MODEL_PATH_ENV = {
    "sd_1_4": "STABLE_DIFFUSION_1_4_PATH",
    "sd_1_5": "STABLE_DIFFUSION_1_5_PATH",
}
CLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)


def _resize_for_clip(images):
    try:
        return nn.interpolate(images, size=(224, 224), mode="bicubic", align_corners=False)
    except TypeError:
        try:
            return nn.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
        except TypeError:
            return nn.interpolate(images, size=(224, 224))


def clip_preprocess(images):
    images = _resize_for_clip(images)
    mean = jt.array(CLIP_IMAGE_MEAN).reshape(1, 3, 1, 1).cast(images.dtype)
    std = jt.array(CLIP_IMAGE_STD).reshape(1, 3, 1, 1).cast(images.dtype)
    return (images - mean) / std


class CLIPTokenImageEncoder:
    def __init__(self, model):
        self.model = model
        if hasattr(self.model, "eval"):
            self.model.eval()

    @jt.no_grad()
    def encode_image(self, images):
        outputs = self.model(pixel_values=images, return_dict=True)
        tokens = outputs.last_hidden_state
        return self.model.vision_model.post_layernorm(tokens)


def resolve_model_id(sd_version: str) -> str:
    if sd_version not in MODEL_IDS:
        raise ValueError(f"Unsupported stable diffusion version: {sd_version}")
    return os.environ.get(MODEL_PATH_ENV[sd_version], MODEL_IDS[sd_version])


def load_model(sd_version="sd_1_5", scheduler_config=None, token="", local_files_only=True):
    from diffusers import DDIMScheduler
    from JDiffusion import StableDiffusionPipeline
    from transformers import CLIPVisionModel

    scheduler = DDIMScheduler(**(scheduler_config or {}))
    pipe = StableDiffusionPipeline.from_pretrained(
        resolve_model_id(sd_version),
        scheduler=scheduler,
        dtype=jt.float32,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
        local_files_only=local_files_only,
        use_auth_token=token or None,
    )

    clip_vision = CLIPVisionModel.from_pretrained(
        "openai/clip-vit-base-patch16",
        local_files_only=local_files_only,
    )
    pipe.clip_model = CLIPTokenImageEncoder(clip_vision)
    pipe.clip_preprocess = clip_preprocess
    return pipe

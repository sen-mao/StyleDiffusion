import importlib


def load_backend_model(backend_module, sd_version, scheduler_config, token="", local_files_only=True):
    """Load a Jittor Stable Diffusion backend.

    The backend module must expose one of:
    - load_model(sd_version=..., scheduler_config=..., token=..., local_files_only=...)
    - StableDiffusionPipeline.from_pretrained(...)

    Returned object is expected to provide the same runtime surface used by the
    original StyleDiffusion algorithm: tokenizer, text_encoder, unet, vae and
    scheduler. For training/editing it should also expose clip_model and an
    optional clip_preprocess callable for image embeddings.
    """
    if not backend_module:
        raise RuntimeError(
            "No Jittor Stable Diffusion backend configured. Pass "
            "--backend_module <python.module> where the module loads a Jittor "
            "pipeline with tokenizer/text_encoder/unet/vae/scheduler and CLIP "
            "image encoder components."
        )

    module = importlib.import_module(backend_module)
    if hasattr(module, "load_model"):
        return module.load_model(
            sd_version=sd_version,
            scheduler_config=scheduler_config,
            token=token,
            local_files_only=local_files_only,
        )

    if hasattr(module, "StableDiffusionPipeline"):
        model_id = {
            "sd_1_4": "CompVis/stable-diffusion-v1-4",
            "sd_1_5": "runwayml/stable-diffusion-v1-5",
        }.get(sd_version)
        if model_id is None:
            raise ValueError(f"Unsupported stable diffusion version: {sd_version}")
        return module.StableDiffusionPipeline.from_pretrained(
            model_id,
            use_auth_token=token,
            scheduler_config=scheduler_config,
            local_files_only=local_files_only,
        )

    raise RuntimeError(
        f"Backend module {backend_module!r} does not expose load_model() or "
        "StableDiffusionPipeline."
    )

#!/usr/bin/env python3
"""Train SDXL LoRA adapters for pecha paper/ink texture transfer."""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection

from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler

try:
    from diffusers.utils import convert_state_dict_to_diffusers
except ImportError:  # pragma: no cover
    def convert_state_dict_to_diffusers(state_dict):
        return state_dict

try:
    from peft import LoraConfig
    try:
        from peft.utils import get_peft_model_state_dict
    except ImportError:  # pragma: no cover
        from peft import get_peft_model_state_dict
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "peft is required for LoRA training. Install with: pip install peft"
    ) from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tibetan_utils.arg_utils import create_train_texture_lora_parser

LOGGER = logging.getLogger("train_texture_lora_sdxl")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _configure_logging(is_main_process: bool) -> None:
    level = logging.INFO if is_main_process else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _discover_images(dataset_dir: Path) -> List[Path]:
    images_dir = dataset_dir / "images"
    root = images_dir if images_dir.is_dir() else dataset_dir
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


class TextureImageDataset(Dataset):
    def __init__(self, image_paths: List[Path], resolution: int):
        self.image_paths = image_paths
        self.transform = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        with Image.open(path) as image:
            image = image.convert("RGB")
            pixel_values = self.transform(image)
        return {
            "pixel_values": pixel_values,
            "path": str(path),
        }


def _encode_prompt(
    prompt: str,
    tokenizer_one,
    tokenizer_two,
    text_encoder_one,
    text_encoder_two,
    device: torch.device,
):
    prompts = [prompt]
    prompt_embeds_list = []
    pooled_prompt_embeds = None

    for tokenizer, text_encoder in (
        (tokenizer_one, text_encoder_one),
        (tokenizer_two, text_encoder_two),
    ):
        text_inputs = tokenizer(
            prompts,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        encoder_output = text_encoder(text_input_ids, output_hidden_states=True)

        prompt_embeds = encoder_output.hidden_states[-2]
        prompt_embeds_list.append(prompt_embeds)

        # SDXL uses pooled prompt embeddings from text_encoder_2.
        pooled_prompt_embeds = encoder_output[0]

    prompt_embeds = torch.cat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(prompt_embeds.shape[0], -1)

    return prompt_embeds, pooled_prompt_embeds


def _get_add_time_ids(batch_size: int, resolution: int, dtype: torch.dtype, device: torch.device):
    add_time_ids = torch.tensor(
        [[resolution, resolution, 0, 0, resolution, resolution]],
        dtype=dtype,
        device=device,
    )
    return add_time_ids.repeat(batch_size, 1)


def _load_models(base_model_id: str):
    tokenizer_one = AutoTokenizer.from_pretrained(base_model_id, subfolder="tokenizer", use_fast=False)
    tokenizer_two = AutoTokenizer.from_pretrained(base_model_id, subfolder="tokenizer_2", use_fast=False)

    text_encoder_one = CLIPTextModel.from_pretrained(base_model_id, subfolder="text_encoder")
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(base_model_id, subfolder="text_encoder_2")

    vae = AutoencoderKL.from_pretrained(base_model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(base_model_id, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(base_model_id, subfolder="scheduler")

    return tokenizer_one, tokenizer_two, text_encoder_one, text_encoder_two, vae, unet, noise_scheduler


def _save_lora_weights(
    accelerator: Accelerator,
    args,
    unet,
    text_encoder_one,
    text_encoder_two,
) -> Path:
    unwrapped_unet = accelerator.unwrap_model(unet)
    unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet))

    text_encoder_lora_state_dict = None
    text_encoder_2_lora_state_dict = None
    if args.train_text_encoder:
        unwrapped_te_one = accelerator.unwrap_model(text_encoder_one)
        unwrapped_te_two = accelerator.unwrap_model(text_encoder_two)
        text_encoder_lora_state_dict = convert_state_dict_to_diffusers(
            get_peft_model_state_dict(unwrapped_te_one)
        )
        text_encoder_2_lora_state_dict = convert_state_dict_to_diffusers(
            get_peft_model_state_dict(unwrapped_te_two)
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    StableDiffusionXLPipeline.save_lora_weights(
        save_directory=str(output_dir),
        unet_lora_layers=unet_lora_state_dict,
        text_encoder_lora_layers=text_encoder_lora_state_dict,
        text_encoder_2_lora_layers=text_encoder_2_lora_state_dict,
        weight_name=args.lora_weights_name,
        safe_serialization=True,
    )
    return output_dir / args.lora_weights_name


def run(args) -> dict:
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    _configure_logging(accelerator.is_main_process)

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if args.max_train_steps <= 0:
        raise ValueError("max_train_steps must be > 0")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if args.rank <= 0:
        raise ValueError("rank must be > 0")

    set_seed(args.seed)

    image_paths = _discover_images(dataset_dir)
    if not image_paths:
        raise RuntimeError(f"No training images found in {dataset_dir}")
    LOGGER.info("Found %d images for LoRA training", len(image_paths))

    (
        tokenizer_one,
        tokenizer_two,
        text_encoder_one,
        text_encoder_two,
        vae,
        unet,
        noise_scheduler,
    ) = _load_models(args.base_model_id)

    if not hasattr(unet, "add_adapter"):
        raise RuntimeError(
            "UNet.add_adapter() is unavailable. Update diffusers to a version with PEFT integration."
        )

    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    unet.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    unet.add_adapter(unet_lora_config)

    if args.train_text_encoder:
        text_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.lora_alpha,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
        text_encoder_one.add_adapter(text_lora_config)
        text_encoder_two.add_adapter(text_lora_config)

    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    if args.train_text_encoder:
        trainable_params.extend(p for p in text_encoder_one.parameters() if p.requires_grad)
        trainable_params.extend(p for p in text_encoder_two.parameters() if p.requires_grad)

    if not trainable_params:
        raise RuntimeError("No trainable parameters found for LoRA optimization")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )

    dataset = TextureImageDataset(image_paths=image_paths, resolution=args.resolution)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=max(args.num_workers, 0),
        pin_memory=True,
    )

    steps_per_epoch = max(1, math.ceil(len(dataset) / args.batch_size))
    num_train_epochs = max(1, math.ceil(args.max_train_steps / steps_per_epoch))

    lr_scheduler = get_scheduler(
        name="constant",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=args.max_train_steps,
    )

    if args.train_text_encoder:
        unet, text_encoder_one, text_encoder_two, optimizer, dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder_one, text_encoder_two, optimizer, dataloader, lr_scheduler
        )
    else:
        unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, dataloader, lr_scheduler
        )

    params_to_clip = [p for p in unet.parameters() if p.requires_grad]
    if args.train_text_encoder:
        params_to_clip.extend(p for p in text_encoder_one.parameters() if p.requires_grad)
        params_to_clip.extend(p for p in text_encoder_two.parameters() if p.requires_grad)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)

    if not args.train_text_encoder:
        text_encoder_one.to(accelerator.device, dtype=weight_dtype)
        text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    vae.eval()
    if not args.train_text_encoder:
        text_encoder_one.eval()
        text_encoder_two.eval()

    if not args.train_text_encoder:
        with torch.no_grad():
            cached_prompt_embeds, cached_pooled_prompt_embeds = _encode_prompt(
                prompt=args.prompt,
                tokenizer_one=tokenizer_one,
                tokenizer_two=tokenizer_two,
                text_encoder_one=text_encoder_one,
                text_encoder_two=text_encoder_two,
                device=accelerator.device,
            )
            cached_prompt_embeds = cached_prompt_embeds.to(dtype=weight_dtype)
            cached_pooled_prompt_embeds = cached_pooled_prompt_embeds.to(dtype=weight_dtype)
    else:
        cached_prompt_embeds = None
        cached_pooled_prompt_embeds = None

    progress_bar = tqdm(
        range(args.max_train_steps),
        disable=not accelerator.is_local_main_process,
        desc="train",
    )

    global_step = 0
    unet.train()
    if args.train_text_encoder:
        text_encoder_one.train()
        text_encoder_two.train()

    for epoch in range(num_train_epochs):
        for batch in dataloader:
            if global_step >= args.max_train_steps:
                break

            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)

                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=latents.device,
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                if args.train_text_encoder:
                    prompt_embeds, pooled_prompt_embeds = _encode_prompt(
                        prompt=args.prompt,
                        tokenizer_one=tokenizer_one,
                        tokenizer_two=tokenizer_two,
                        text_encoder_one=text_encoder_one,
                        text_encoder_two=text_encoder_two,
                        device=accelerator.device,
                    )
                    prompt_embeds = prompt_embeds.to(dtype=weight_dtype)
                    pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=weight_dtype)
                else:
                    prompt_embeds = cached_prompt_embeds.repeat(latents.shape[0], 1, 1)
                    pooled_prompt_embeds = cached_pooled_prompt_embeds.repeat(latents.shape[0], 1)

                add_time_ids = _get_add_time_ids(
                    batch_size=latents.shape[0],
                    resolution=args.resolution,
                    dtype=prompt_embeds.dtype,
                    device=latents.device,
                )

                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs={
                        "text_embeds": pooled_prompt_embeds,
                        "time_ids": add_time_ids,
                    },
                ).sample

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unsupported prediction type: {noise_scheduler.config.prediction_type}"
                    )

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_clip, max_norm=1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix(
                    loss=f"{loss.detach().float().item():.4f}",
                    lr=f"{lr_scheduler.get_last_lr()[0]:.2e}",
                    epoch=str(epoch + 1),
                )

        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()

    saved_lora_path = None
    training_config_path = output_dir / "training_config.json"

    if accelerator.is_main_process:
        saved_lora_path = _save_lora_weights(
            accelerator=accelerator,
            args=args,
            unet=unet,
            text_encoder_one=text_encoder_one,
            text_encoder_two=text_encoder_two,
        )

        training_config = {
            "base_model_id": args.base_model_id,
            "dataset_dir": str(dataset_dir),
            "num_training_images": len(image_paths),
            "resolution": args.resolution,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "max_train_steps": args.max_train_steps,
            "rank": args.rank,
            "lora_alpha": args.lora_alpha,
            "mixed_precision": args.mixed_precision,
            "gradient_checkpointing": bool(args.gradient_checkpointing),
            "train_text_encoder": bool(args.train_text_encoder),
            "prompt": args.prompt,
            "seed": args.seed,
            "lora_weights": str(saved_lora_path),
        }
        with training_config_path.open("w", encoding="utf-8") as f:
            json.dump(training_config, f, indent=2, ensure_ascii=True)

        LOGGER.info("Saved LoRA weights to %s", saved_lora_path)
        LOGGER.info("Saved training config to %s", training_config_path)

    accelerator.end_training()

    return {
        "steps": global_step,
        "output_dir": str(output_dir),
        "lora_path": str(saved_lora_path) if saved_lora_path else "",
        "training_config": str(training_config_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = create_train_texture_lora_parser()
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

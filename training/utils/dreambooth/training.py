#@title Training function
from typing import Union
import accelerate
import diffusers
from tqdm import tqdm
from pathlib import Path
from .dataset import PromptDataset, DreamBoothDataset
import torch
import gc
import bitsandbytes as bnb
import itertools
import glob
import os
import math
import torch.nn.functional as F


def generate_class_images(
    prior_preservation: bool, 
    class_data_root: Union[str, Path], 
    num_class_images: int, 
    pretrained_model_name_or_path: Union[str, Path], 
    class_prompt: str, 
    sample_batch_size: int
):
    if (prior_preservation):
        print("Prior preservation is turned on. Generating class images...")
        class_images_dir = Path(class_data_root)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < num_class_images:
            pipeline = diffusers.StableDiffusionPipeline.from_pretrained(
                pretrained_model_name_or_path, revision="fp16", torch_dtype=torch.float16
            ).to("cuda")
            pipeline.enable_attention_slicing()
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = num_class_images - cur_class_images
            print(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=sample_batch_size)

            for example in tqdm(sample_dataloader, desc="Generating class images"):
                images = pipeline(example["prompt"]).images

                for i, image in enumerate(images):
                    image.save(class_images_dir / f"{example['index'][i] + cur_class_images}.jpg")
            pipeline = None
            gc.collect()
            del pipeline
            with torch.no_grad():
                torch.cuda.empty_cache()
    else:
        print("Prior preservation loss is turned off. Skip...")




def training_function(cfg, text_encoder, vae, unet, tokenizer):
    logger = accelerate.logging.get_logger(__name__)

    accelerate.utils.set_seed(cfg.seed)

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
    )

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if cfg.train_text_encoder and cfg.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    vae.requires_grad_(False)
    if not cfg.train_text_encoder:
        text_encoder.requires_grad_(False)

    if cfg.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if cfg.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if cfg.use_8bit_adam:
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    params_to_optimize = (
        itertools.chain(unet.parameters(), text_encoder.parameters()) if cfg.train_text_encoder else unet.parameters()
    )

    optimizer = optimizer_class(
        params_to_optimize,
        lr=cfg.learning_rate,
    )

    noise_scheduler = diffusers.DDPMScheduler.from_config(cfg.pretrained_model_name_or_path, subfolder="scheduler")

    train_dataset = DreamBoothDataset(
        instance_data_root=cfg.instance_data_dir,
        instance_prompt=cfg.instance_prompt,
        class_data_root=cfg.class_data_dir if cfg.with_prior_preservation else None,
        class_prompt=cfg.class_prompt,
        tokenizer=tokenizer,
        size=cfg.resolution,
        center_crop=cfg.center_crop,
    )

    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]

        # concat class and instance examples for prior preservation
        if cfg.with_prior_preservation:
            input_ids += [example["class_prompt_ids"] for example in examples]
            pixel_values += [example["class_images"] for example in examples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = tokenizer.pad(
            {"input_ids": input_ids},
            padding="max_length",
            return_tensors="pt",
            max_length=tokenizer.model_max_length
        ).input_ids

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }
        return batch

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.train_batch_size, shuffle=True, collate_fn=collate_fn
    )

    lr_scheduler = diffusers.optimization.get_scheduler(
        cfg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps * cfg.gradient_accumulation_steps,
        num_training_steps=cfg.max_train_steps * cfg.gradient_accumulation_steps,
    )

    if cfg.train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    vae.to(accelerator.device, dtype=weight_dtype)
    vae.decoder.to("cpu")
    if not cfg.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
    num_train_epochs = math.ceil(cfg.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = cfg.train_batch_size * accelerator.num_processes * cfg.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Instantaneous batch size per device = {cfg.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {cfg.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {cfg.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(cfg.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0

    for epoch in range(num_train_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict the noise residual
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if cfg.with_prior_preservation:
                    # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
                    noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)

                    # Compute instance loss
                    loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none").mean([1, 2, 3]).mean()

                    # Compute prior loss
                    prior_loss = F.mse_loss(noise_pred_prior.float(), target_prior.float(), reduction="mean")

                    # Add the prior loss to the instance loss.
                    loss = loss + cfg.prior_loss_weight * prior_loss
                else:
                    loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet.parameters(), text_encoder.parameters())
                        if cfg.train_text_encoder
                        else unet.parameters()
                    )
                    accelerator.clip_grad_norm_(unet.parameters(), cfg.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % cfg.save_steps == 0:
                    if accelerator.is_main_process:
                        from natsort import natsorted 
                        import shutil

                        all_ckpts = natsorted(glob.glob(cfg.output_dir + "/*"))
                        residual_ckpts = []
                        if len(all_ckpts):
                            residual_ckpts = list(set(all_ckpts) - {all_ckpts[-1]})

                        for ckpt in residual_ckpts:
                            print(f"Deleting residual ckpt: {ckpt}")
                            shutil.rmtree(ckpt)
                        pipeline = diffusers.StableDiffusionPipeline.from_pretrained(
                            cfg.pretrained_model_name_or_path,
                            unet=accelerator.unwrap_model(unet),
                            text_encoder=accelerator.unwrap_model(text_encoder),
                        )
                        save_path = os.path.join(cfg.output_dir, f"checkpoint-{global_step}")
                        pipeline.save_pretrained(save_path)

            logs = {"loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)

            if global_step >= cfg.max_train_steps:
                break

        accelerator.wait_for_everyone()

    # Create the pipeline using using the trained modules and save it.
    if accelerator.is_main_process:
        pipeline = diffusers.StableDiffusionPipeline.from_pretrained(
            cfg.pretrained_model_name_or_path,
            unet=accelerator.unwrap_model(unet),
            text_encoder=accelerator.unwrap_model(text_encoder),
        )
        pipeline.save_pretrained(cfg.output_dir)
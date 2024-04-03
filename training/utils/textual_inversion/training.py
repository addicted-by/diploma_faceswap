#@title Training function
import accelerate
import torch 
from .dataset import create_dataloader, TextualInversionDataset
from .utils import freeze_params
import itertools
import math
import bitsandbytes as bnb
import os
from tqdm import tqdm
import torch.nn.functional as F
import diffusers


def save_progress(
    text_encoder, 
    placeholder_token_id,
    placeholder_token, 
    accelerator, 
    save_path
):
    logger = accelerate.logging.get_logger()
    logger.info("Saving embeddings")
    learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[placeholder_token_id]
    learned_embeds_dict = {placeholder_token: learned_embeds.detach().cpu()}
    torch.save(learned_embeds_dict, save_path)



def training_function(cfg, text_encoder, vae, unet, tokenizer):
    logger = accelerate.logging.get_logger(__name__)

    train_dataset = TextualInversionDataset(
        instance_data_root=cfg.instance_data_dir,
        tokenizer=tokenizer,
        learnable_property=cfg.property, #Option selected above between object and style
        size=vae.sample_size,
        repeats=cfg.repeats,
        set="train",
        placeholder_token=cfg.placeholder_token,
        center_crop=cfg.center_crop,
    )
    # Freeze vae and unet
    freeze_params(vae.parameters())
    freeze_params(unet.parameters())
    # Freeze all parameters except for the token embeddings in text encoder
    params_to_freeze = itertools.chain(
        text_encoder.text_model.encoder.parameters(),
        text_encoder.text_model.final_layer_norm.parameters(),
        text_encoder.text_model.embeddings.position_embedding.parameters(),
    )
    freeze_params(params_to_freeze)

    # train_batch_size = cfg.train_batch_size #["train_batch_size"]
    # gradient_accumulation_steps = cfg["gradient_accumulation_steps"]
    # learning_rate = cfg["learning_rate"]
    # max_train_steps = cfg["max_train_steps"]
    # output_dir = cfg["output_dir"]
    # gradient_checkpointing = cfg["gradient_checkpointing"]

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision
    )

    if cfg.gradient_checkpointing:
        text_encoder.gradient_checkpointing_enable()
        unet.enable_gradient_checkpointing()

    train_dataloader = create_dataloader(train_dataset, cfg.train_batch_size)

    if cfg.scale_lr:
        learning_rate = (
            cfg.learning_rate * cfg.gradient_accumulation_steps * cfg.train_batch_size * accelerator.num_processes
        )
    else:
        learning_rate = cfg.learning_rate

    if cfg.use_8bit_adam:
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW
    # Initialize the optimizer
    optimizer = optimizer_class(
        text_encoder.get_input_embeddings().parameters(),  # only optimize the embeddings
        lr=learning_rate,
    )

    noise_scheduler = diffusers.DDPMScheduler.from_config(cfg.pretrained_model_name_or_path, subfolder="scheduler")

    text_encoder, optimizer, train_dataloader = accelerator.prepare(
        text_encoder, optimizer, train_dataloader
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and unet to device
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)

    # Keep vae in eval mode as we don't train it
    vae.eval()
    # Keep unet in train mode to enable gradient checkpointing
    unet.train()

    
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
        text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(text_encoder):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample().detach()
                latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device).long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict the noise residual
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states.to(weight_dtype)).sample

                 # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(noise_pred, target, reduction="none").mean([1, 2, 3]).mean()
                accelerator.backward(loss)

                # Zero out the gradients for all token embeddings except the newly added
                # embeddings for the concept, as we only want to optimize the concept embeddings
                if accelerator.num_processes > 1:
                    grads = text_encoder.module.get_input_embeddings().weight.grad
                else:
                    grads = text_encoder.get_input_embeddings().weight.grad
                # Get the index for tokens that we want to zero the grads for
                index_grads_to_zero = torch.arange(len(tokenizer)) != cfg.placeholder_token_id
                grads.data[index_grads_to_zero, :] = grads.data[index_grads_to_zero, :].fill_(0)

                optimizer.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if global_step % cfg.save_steps == 0:
                    save_path = os.path.join(cfg.output_dir, f"learned_embeds-step-{global_step}.bin")
                    save_progress(
                        text_encoder, 
                        cfg.placeholder_token_id, 
                        cfg.placeholder_token, 
                        accelerator, 
                        save_path
                    )

            logs = {"loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)

            if global_step >= cfg.max_train_steps:
                break

        accelerator.wait_for_everyone()


    # Create the pipeline using using the trained modules and save it.
    if accelerator.is_main_process:
        pipeline = diffusers.StableDiffusionPipeline.from_pretrained(
            cfg.pretrained_model_name_or_path,
            text_encoder=accelerator.unwrap_model(text_encoder),
            tokenizer=tokenizer,
            vae=vae,
            unet=unet,
        )
        pipeline.save_pretrained(cfg.output_dir)
        # Also save the newly trained embeddings
        save_path = os.path.join(cfg.output_dir, f"learned_embeds.bin")
        save_progress(text_encoder, cfg.placeholder_token_id, accelerator, save_path)
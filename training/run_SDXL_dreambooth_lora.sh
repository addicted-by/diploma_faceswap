export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export INSTANCE_DIR="data/scarlett"
export OUTPUT_DIR="lora-trained-xl-scarlett"
export VAE_PATH="madebyollin/sdxl-vae-fp16-fix"

accelerate launch train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --pretrained_vae_model_name_or_path=$VAE_PATH \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="fp16" \
  --instance_prompt="a photo of sks woman" \
  --resolution=256 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-6 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="A photo of sks woman in a Paris" \
  --validation_epochs=25 \
  --seed="0" \
  --push_to_hub

#!/usr/bin/bash
# source /etc/bashrc
# source /etc/profile
# source /etc/profile.d/modules.sh
# module load gcc/9.2.0

# source ~/.bashrc
# source ~/.bash_profile
# echo JOB STARTED
# nvidia-smi
# source /home/changl25/miniconda3/etc/profile.d/conda.sh
# conda activate ziplora
# cd /home/changl25/ziplora-pytorch




export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"


# Final setting
# overlap_first_flag=True 
# accumulate_cone_flag=True
# column_per_sep_flag=False
# finetune_mask_flag=False 
# export period_sample_epoch=3
# export sampled_column_ratio=0.1

# For weight similarity
export CONTENT_LR=0.00005
export STYLE_LR=0.00005
export weight_lr=0.005
export similarity_lambda=0.01
export RANK=64
export WANDB_NAME="unziplora"
export INSTANCE_DIR="instance_data/anime_cat"
export OUTPUT_DIR="models"
export STEPS=1000

# Training prompt
export PROMPT="A monadikos cat in anime illustration style"
export CONTENT_FORWARD_PROMPT="A monadikos cat"
export STYLE_FORWARD_PROMPT="A cat in anime illustration style"
# For validation
export VALID_CONTENT="A monadikos cat standing on a table"
export VALID_PROMPT="A monadikos cat standing on a table in anime illustration style"
export VALID_STYLE="A cat in anime illustration style standing on a table"

# for content validation
export VALID_CONTENT_PROMPT="a photo of a monadikos cat standing on a table"

# for style validation
export VALID_STYLE_PROMPT="A dog in anime illustration style"



accelerate launch train_inverse_ziplora_layer_column.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --name=$WANDB_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="${PROMPT}" \
  --content_forward_prompt="${CONTENT_FORWARD_PROMPT}" \
  --style_forward_prompt="${STYLE_FORWARD_PROMPT}" \
  --rank="${RANK}" \
  --resolution=1024 \
  --train_batch_size=1 \
  --content_learning_rate="${CONTENT_LR}" \
  --style_learning_rate="${STYLE_LR}" \
  --weight_learning_rate="$weight_lr" \
  --similarity_lambda="$similarity_lambda" \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps="$STEPS" \
  --checkpointing_steps=500 \
  --validation_epochs=$validate_epochs \
  --mixed_precision="fp16" \
  --seed="0" \
  --use_8bit_adam \
  --push_to_hub \
  --validation_content="${VALID_CONTENT}" \
  --validation_style="${VALID_STYLE}" \
  --validation_prompt="${VALID_PROMPT}" \
  --validation_prompt_style="${VALID_STYLE_PROMPT}" \
  --validation_prompt_content="${VALID_CONTENT_PROMPT}"
  # --with_period_column_separation=$column_per_sep_flag \
  # --sample_times=$period_sample_epoch \
  # --column_ratio=$sampled_column_ratio \
  # --with_no_overlap_first=$overlap_first_flag \
  # --with_accumulate_cone=$accumulate_cone_flag \
  # --with_finetune_mask=$finetune_mask_flag \
  # --with_saved_per_validation="${saved_per_validation_flag}" \
  # --with_image_per_validation="${image_per_validation_flag}" \
  # --with_grad_record="$grad_record_flag" \

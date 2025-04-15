export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"

# Hyper parameters
export period_sample_epoch=3
export sampled_column_ratio=0.1
freeze_flag=True
column_per_sep_flag=True

# For weight similarity
export CONTENT_LR=0.00005
export STYLE_LR=0.00005
export weight_lr=0.005
export similarity_lambda=0.01
export RANK=64
export WANDB_NAME="unziplora"
export INSTANCE_DIR="instance_data/pop_rose"
export OUTPUT_DIR="models"
export STEPS=600

# Training prompt
export PROMPT="A monadikos rose in pop art style"
export CONTENT_FORWARD_PROMPT="A monadikos rose"
export STYLE_FORWARD_PROMPT="A rose in pop art style"
# For validation
export VALID_CONTENT="A monadikos rose on a table"
export VALID_PROMPT="A monadikos rose on a table in pop art style"
export VALID_STYLE="A rose in pop art style on a table"

# for content validation
export VALID_CONTENT_PROMPT="a photo of a monadikos rose on a table"

# for style validation
export VALID_STYLE_PROMPT="A dog in pop art style"



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
  --validation_prompt_content="${VALID_CONTENT_PROMPT}" \
  --sample_times=$period_sample_epoch \
  --column_ratio=$sampled_column_ratio \
  --with_freeze_unet="${freeze_flag}" \
  --with_period_column_separation=$column_per_sep_flag \

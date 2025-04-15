#!/usr/bin/bash
source /etc/bashrc
source /etc/profile
source /etc/profile.d/modules.sh
module load gcc/9.2.0

source ~/.bashrc
source ~/.bash_profile
echo JOB STARTED
nvidia-smi
# python -c "print('Script python start')"

export HOME=/home/changl25/
export DATAROOT=/data/changl25

source /home/changl25/miniconda3/etc/profile.d/conda.sh
conda activate ziplora

export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"

INPUT_GROUP=("teapot" "stuffed_waterpaint" "cartoon_dog" "fox_waterpaint" \
"watercolor_horse" "sketch_cat" "anime_cat" "chinese_deer"  \
 "kid_draw_teapot" "waterpaint_dog")
CHOSEN_NUM=4
chosen_value=${INPUT_GROUP[$CHOSEN_NUM % 10]}

saved_per_validation_flag=False
image_per_validation_flag=False
grad_record_flag=True

freeze_flag=False
lora_sep_flag=False
column_per_sep_flag=False

style_flag=False
norm_flag=False
normalized_flag=False
color_flag=False
style_forward_flag=False
content_forward_flag=False
sample_flag=False
same_seed_flag=True 

finetune_mask_flag=False 
orthog_init_flag=False
one_shot_flag=False
# Final setting
overlap_first_flag=True 
accumulate_cone_flag=True

overlap_first_flag=False 
accumulate_cone_flag=False

# For weight similarity
export ORTHOG_WEIGHT=0.5
# For prior preservation setting
export PRIOR_WEIGHT=0.5
export PRIOR_WEIGHT_2=0.5
export CONTENT_LR=0.00005
export STYLE_LR=0.00005
export weight_lr=0.005
export RANK=64
export style_weight=0.1
export norm_weight=0.00005
export WANDB_NAME="inverse_ziplora_col_ablation"
export OUTPUT_DIR_BASE="/data/changl25/inverse_ziplora/inverse_ziplora_only_lora"
# export OUTPUT_DIR_BASE="/home/changl25/inverse_ziplora_final/input_res"
export OUTPUT_DIR_TAIL="sampled_num_3_0.1"
export TAG="ablation"
export STEPS=1000
export validate_epochs=100
export period_sample_epoch=3
export sampled_column_ratio=0.1


accelerate launch train_inverse_ziplora_layer_column.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --name=$WANDB_NAME \
  --tag $TAG $TAG2 \
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
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps="$STEPS" \
  --checkpointing_steps=1024 \
  --validation_epochs=$validate_epochs \
  --mixed_precision="fp16" \
  --seed="0" \
  --with_same_seed="$same_seed_flag" \
  --use_8bit_adam \
  --push_to_hub \
  --validation_content="${VALID_CONTENT}" \
  --validation_style="${VALID_STYLE}" \
  --validation_prompt="${VALID_PROMPT}" \
  --validation_prompt_style="${VALID_STYLE_PROMPT}" \
  --validation_prompt_content="${VALID_CONTENT_PROMPT}" \
  --num_class_images=200 \
  --with_lora_separation=$lora_sep_flag \
  --with_period_column_separation=$column_per_sep_flag \
  --sample_times=$period_sample_epoch \
  --column_ratio=$sampled_column_ratio \
  --with_no_overlap_first=$overlap_first_flag \
  --with_accumulate_cone=$accumulate_cone_flag \
  --with_finetune_mask=$finetune_mask_flag \
  --similarity_lambda="${ORTHOG_WEIGHT}" \
  --with_style_loss=$style_flag \
  --lam_style=$style_weight \
  --with_saved_per_validation="${saved_per_validation_flag}" \
  --with_image_per_validation="${image_per_validation_flag}" \
  --with_orthog_init="${orthog_init_flag}" \
  --with_one_shot="${one_shot_flag}" \
  --with_normalization="${norm_flag}" \
  --with_per_step_normalization="${normalized_flag}" \
  --lam_norm="${norm_weight}" \
  --with_grad_record="$grad_record_flag" \
  --with_style_forward="$style_forward_flag" \
  --with_content_forward="$content_forward_flag" \

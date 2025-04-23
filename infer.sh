export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export RANK=64
export NUM=1

export OUTPUT_DIR="models/pop_art_rose/pop_art_rose"
export SAVE_DIR="output/pop_art_rose/"

export VALID_PROMPTS=(
  "a monadikos rose on a skateboard in pop art style"
  "a monadikos rose in a snowy landscape in pop art style"
)
VALID_PROMPT=$(IFS=,; echo "${VALID_PROMPTS[*]}")
export VALID_PROMPT

export VALID_STYLES=(
  "a rose on a skateboard in pop art style"
  "a rose in a snowy landscape in pop art style"
)
VALID_STYLE=$(IFS=,; echo "${VALID_STYLES[*]}")
export VALID_STYLE

export VALID_CONTENTS=(
  "a monadikos rose on a skateboard"
  "a monadikos rose in a snowy landscape"
)
VALID_CONTENT=$(IFS=,; echo "${VALID_CONTENTS[*]}")
export VALID_CONTENT

export VALID_CONTENT_RECON_PROMPTS=(
  "A photo of monadikos rose on a table"
  "A photo of monadikos rose in a beach"
)
VALID_CONTENT_RECON_PROMPT=$(IFS=,; echo "${VALID_CONTENT_RECON_PROMPTS[*]}")
export VALID_CONTENT_RECON_PROMPT

export VALID_STYLE_PROMPTS=(
  "A dog in pop art style"
  "A chair in pop art style"
)
VALID_STYLE_PROMPT=$(IFS=,; echo "${VALID_STYLE_PROMPTS[*]}")
export VALID_STYLE_PROMPT

accelerate launch infer.py \
  --output_dir="$OUTPUT_DIR" \
  --rank="${RANK}" \
  --num="${NUM}" \
  --with_unziplora \
  --save_dir="$SAVE_DIR" \
  --validation_prompt_content_recontext="${VALID_CONTENT_RECON_PROMPT}" \
  --validation_prompt_style="${VALID_STYLE_PROMPT}" \
  --validation_prompt="${VALID_PROMPT}" \
  --validation_prompt_style_forward="${VALID_STYLE}" \
  --validation_prompt_content_forward="${VALID_CONTENT}" 


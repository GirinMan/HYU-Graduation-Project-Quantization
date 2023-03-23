#!/bin/bash
wandb login --relogin
export WANDB_LOG_MODEL="false"
export WANDB_PROJECT="template_based_SFT_peft_bitsandbytes"
export WANDB_WATCH="false"
python finetune_peft_classification.py \
configs/imdb_classification_config.json
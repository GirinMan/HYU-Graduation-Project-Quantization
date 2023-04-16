#!/bin/bash
export WANDB_LOG_MODEL="false"
export WANDB_PROJECT="template_based_SFT_peft_bitsandbytes"
export WANDB_WATCH="false"
python finetune_peft_generation.py configs/opt-1.3b_samsum_8bit.json

#!/bin/bash
#LOAD='MAGAer13/mplug-owl2-llama2-7b'
LOAD='path/to/pretrain'
DATA_FILE=path/to/json_data
Image_root='/data/dataset/AVA_dataset/images/images'

model_path=path/to/save
deepspeed mplug_owl2/train/train_mem.py \
    --min_score 1 \
    --max_score 10 \
    --num_tokens 10 \
    --l1_weight 0 \
    --ce_weight 10 \
    --emd_weight 0 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path $LOAD \
    --version v1 \
    --data_path $DATA_FILE \
    --image_folder $Image_root \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $model_path \
    --num_train_epochs 2 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --tune_visual_abstractor True \
    --freeze_vision_model False \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard


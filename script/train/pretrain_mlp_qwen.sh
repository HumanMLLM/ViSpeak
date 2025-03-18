#!/bin/bash

MODEL_TYPE=qwen2p5_instruct
OUTPUT_DIR=$1
OUTPUT_DIR_FT=${OUTPUT_DIR}/llava-s1-pretrain_mlp
mkdir -p ${OUTPUT_DIR_FT}

deepspeed --master_port=29505 --include localhost:0,1,3,4,5,7 vispeak/train/train.py \
    --deepspeed ./script/deepspeed/zero2.json \
    --model_name_or_path /mnt/data/shenghao/models/VITA-1.5 \
    --model_type $MODEL_TYPE \
    --version qwen2p5_instruct \
    --dataset_use pretrain_offline_omni_data \
    --vision_tower /mnt/data/shenghao/models/InternViT-300M-448px \
    --mm_projector_type mlp2x_gelu \
    --combination linear_combine \
    --tune_mm_mlp_adapter True \
    --tune_audio_mlp_adapter True \
    --audio_encoder /mnt/data/shenghao/models/VITA-1.5/audio-encoder-Qwen2-7B-1107-weight-base-11wh-tunning \
    --freeze_audio_encoder True \
    --image_aspect_ratio square \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir ${OUTPUT_DIR_FT} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 5e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 6200 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    2>&1 | tee -a ${OUTPUT_DIR_FT}/log.txt && echo "Done."





#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`


export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
#export NCCL_IB_HCA=mlx5_2:1,mlx5_2:1
#export NCCL_IB_SL=3
#export NCCL_CHECKS_DISABLE=1
export NCCL_P2P_DISABLE=0
#export NCCL_LL_THRESHOLD=16384
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_DEBUG=INFO
export PYTHONPATH=./
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:3950"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

export NCCL_TIMEOUT=25200
MODEL_TYPE=qwen2p5_instruct
OUTPUT_DIR=output
OUTPUT_DIR_FT=${OUTPUT_DIR}/llava-s1-finetune_video_lora
mkdir -p ${OUTPUT_DIR_FT}

torchrun  --nproc_per_node 8 --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
     vispeak/train/train.py \
    --deepspeed ./script/deepspeed/zero2.json \
    --model_name_or_path /mnt/data/shenghao/models/VITA-1.5 \
    --model_type $MODEL_TYPE \
    --version qwen2p5_instruct \
    --dataset_use fintune_offline_omni_data \
    --vision_tower /mnt/data/shenghao/models/InternViT-300M-448px \
    --pretrain_mm_mlp_adapter ${OUTPUT_DIR}/llava-s1-pretrain_mlp/checkpoint-20317/mm_projector.bin \
    --pretrain_audio_mlp_adapter ${OUTPUT_DIR}/llava-s1-pretrain_mlp/checkpoint-20317/audio_adpter.bin \
    --pretrain_combination_modules ${OUTPUT_DIR}/llava-s1-pretrain_mlp/checkpoint-20317/combination_params.bin \
    --mm_projector_type mlp2x_gelu \
    --combination adaptive_combine \
    --audio_encoder /mnt/data/shenghao/models/VITA-1.5/audio-encoder-Qwen2-7B-1107-weight-base-11wh-tunning \
    --freeze_audio_encoder True \
    --freeze_audio_encoder_adapter True \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 1e-5 \
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
    --save_steps 1000 \
    --save_total_limit 20 \
    --learning_rate 1e-4 \
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
    2>&1 | tee -a ${OUTPUT_DIR_FT}/log_node_$RANK.txt && echo "Done."






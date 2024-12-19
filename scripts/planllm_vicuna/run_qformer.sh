#!/bin/bash
# Please modify the ${MASTER_NODE}:${MASTER_PORT}
MASTER_NODE=127.0.0.1 
MASTER_PORT=$((10000 + $RANDOM % 100))
NNODE=1
NUM_GPUS=1
OUTPUT_DIR="checkpoints"
export CUDA_VISIBLE_DEVICES=1

torchrun --rdzv_endpoint=${MASTER_NODE}:${MASTER_PORT} \
    --nnodes=${NNODE} \
    --nproc_per_node=${NUM_GPUS} \
    --rdzv_backend=c10d \
    tasks/train_qformer.py \
    $(dirname $0)/config_qformer.py \
    output_dir $OUTPUT_DIR\
    distributed True \



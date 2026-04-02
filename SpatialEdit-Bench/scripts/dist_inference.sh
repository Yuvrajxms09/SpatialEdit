#!/bin/bash
export CUDA_LAUNCH_BLOCKING=1


RESOLUTION=1024
CONFIG=your_base_path/SpatialEdit_github/configs/spatialedit_base_config.py
CKPT=your_base_path/XVideo/outputs/sft_edit_512p_mmdit_16b_qwen_wan_space/bench_exp13_sp1_world64/checkpoints/checkpoint-5000/lora
META_FILE=SpatialEdit_Bench_Meta_File.json
BENCH_DATA_DIR=SpatialEdit_Bench_Data

SAVE=SpatialEdit_github/SpatialEdit-Bench/eval_output
echo save_path in ${SAVE}

GPUS_PER_NODE=8
NNODES=1
NODE_RANK=0

torchrun --nnodes $NNODES --nproc_per_node $GPUS_PER_NODE --node_rank $NODE_RANK \
    SpatialEdit-Bench/eval_inference.py \
    --config=$CONFIG \
    --ckpt-path=$CKPT \
    --save-path=$SAVE \
    --num-inference-steps=50 \
    --guidance-scale=5.0 \
    --basesize=${RESOLUTION} \
    --neg-prompt="" \
    --meta-file="$META_FILE" \
    --bench-data-dir="$BENCH_DATA_DIR" \
    --seed=42

echo "All Jobs Done."

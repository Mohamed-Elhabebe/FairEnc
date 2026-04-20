#!/bin/bash
DATASET_DIR=</path/to/Harvard-FairVLMed>
RESULT_DIR=</linearprobe_checkpoint_saving/path>
MODEL_ARCH=vit-b16 # Options: vit-b16 | vit-l14
NUM_EPOCH=1000
BATCH_SIZE=512
EVAL_CHCK_PATH=</path/to/checkpoint_to_apply_linearprobing/.pth_file>

python3 <path/to/FairEnc_codebase>/linear_probe.py \
        --dataset_dir ${DATASET_DIR} \
        --output_dir ${RESULT_DIR} \
        --blr 0.00005 \
        --weight_decay 0.0 \
        --batch_size ${BATCH_SIZE} \
        --epochs ${NUM_EPOCH} \
        --model_arch ${MODEL_ARCH} \
        --finetune_checkpoint ${EVAL_CHCK_PATH}
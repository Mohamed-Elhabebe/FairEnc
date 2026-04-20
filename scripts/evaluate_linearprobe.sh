#!/bin/bash
DATASET_DIR=</path/to/Harvard-FairVLMed>
RESULT_DIR=</test_saving_and_logging/path>
MODEL_ARCH=vit-b16 # Options: vit-b16 | vit-l14
EVAL_CHCK_PATH=</path/to/folder_containing/linearprobe_checkpoint_to_evaluate>

python3 <path/to/FairEnc_codebase>/evaluate_linear_probe.py \
        --dataset_dir ${DATASET_DIR} \
        --result_dir ${RESULT_DIR} \
        --perf_file Test_Results.csv \
        --model_arch ${MODEL_ARCH} \
        --pretrained_weights ${EVAL_CHCK_PATH}/clip_best.pth
#!/bin/bash
DATASET_DIR=</path/to/Harvard-FairVLMed>
RESULT_DIR=</test_saving_and_logging/path>
MODEL_ARCH=vit-b16 # Options: vit-b16 | vit-l14
EVAL_CHCK_PATH=</path/to/checkpoint_to_evaluate/.pth_file>

PERF_FILE=Test_Results.csv

python3 <path/to/FairEnc_codebase>/evaluate.py \
        --dataset_dir ${DATASET_DIR} \
        --result_dir ${RESULT_DIR}/results \
        --perf_file ${PERF_FILE} \
        --model_arch ${MODEL_ARCH} \
        --pretrained_weights ${EVAL_CHCK_PATH}
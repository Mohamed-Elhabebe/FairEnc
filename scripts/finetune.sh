#!/bin/bash
DATASET_DIR=</path/to/Harvard-FairVLMed>
RESULT_DIR=</saving_and_logging/path>
MODEL_ARCH=vit-b16 # Options: vit-b16 | vit-l14
NUM_EPOCH=30
SUMMARIZED_NOTE_FILE=qwen_synthetic_all_notes.csv
LR=1e-05
D_PRED_LR=5e-05
BATCH_SIZE=32

PERF_FILE=CLIP_MI_Reg_Multi_Adv_${MODEL_ARCH}_Best_Results.csv

python3 <path/to/FairEnc_codebase>/finetune.py \
        --seed 777 \
        --dataset_dir ${DATASET_DIR} \
        --result_dir ${RESULT_DIR}/results/glaucoma_CLIP_MI_Reg_Multi_Adv_${MODEL_ARCH} \
        --lr ${LR} \
        --batch_size ${BATCH_SIZE} \
        --num_epochs ${NUM_EPOCH} \
        --perf_file ${PERF_FILE} \
        --model_arch ${MODEL_ARCH} \
        --summarized_note_file ${SUMMARIZED_NOTE_FILE} \
        --attributes race gender ethnicity language \
        --train_dataset_type Mixed_Notes \
        --unbiased_prob 0.5 \
        --adv_lambda 1 \
        --vq_num_embeddings 64 \
        --commitment_cost 0.25 \
        --vq_clip_lambda 10 \
        --use_soft_quantization \
        --non_isolate_encoder \
        --non_detach_codebook_for_probs \
        --vq_text_encoder_optimize \
        --reg_lambda 1 \
        --text_contrastive_lambda 0.01 \
        --text_contrastive_temprature 0.07 \
        --d_pred_lr ${D_PRED_LR} \
        --comet_api_key <your_comet_api_key> \
        --comet_project_name <your_comet_project_name> \
        --comet_experiment_name FairEnc_Finetuning
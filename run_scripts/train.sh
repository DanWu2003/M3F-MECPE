#!/usr/bin/env bash
set -euo pipefail

llm_path="${LLM_PATH:-LLMs/Qwen3-8B}"
bert_path="${BERT_PATH:-LLMs/bert-base-chinese}"
jobid="${JOB_ID:-train/qwen3}"
torchrun --nproc_per_node=1 \
    --master_port=34655 \
    train.py \
    --cfg-path lavis/projects/cap_msvd.yaml \
    --options \
    model.arch blip2_vicuna_instruct \
    model.model_type vicuna7b \
    model.load_finetuned False \
    model.load_pretrained False \
    model.num_query_token 32 \
    model.vit_precision fp32 \
    model.freeze_vit True \
    model.memory_bank_length 20 \
    model.audio_memory_bank_length 20 \
    model.num_frames 40 \
    model.llm_model ${llm_path} \
    model.bert_path ${bert_path} \
    model.use_audio False \
    model.use_video False \
    model.max_txt_len 512 \
    model.max_output_txt_len 125 \
    model.qwen True \
    run.init_lr 5e-4 \
    run.max_epoch 1 \
    run.num_beams 5 \
    run.batch_size_train 1 \
    run.batch_size_eval 1 \
    run.accum_grad_iters 32 \
    run.num_workers 6 \
    run.seed 42 \
    run.evaluate False \
    run.valid_splits "['val']" \
    run.report_metric False \
    run.prefix train \
    run.job_id ${jobid}

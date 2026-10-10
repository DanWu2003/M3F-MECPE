#!/usr/bin/env python3
import subprocess
import sys
import os


def main():
    # 从环境变量或命令行参数读取路径，优先环境变量
    llm_path = os.getenv("LLM_PATH", "LLMs/Qwen3-8B")
    bert_path = os.getenv("BERT_PATH", "LLMs/bert-base-chinese")
    jobid = os.getenv("JOB_ID", "train/1")

    # 支持将脚本所在目录视为项目根
    project_root = os.path.abspath(os.path.dirname(__file__))
    train_py = os.path.join(project_root, "train.py")
    cfg_path = os.path.join(project_root, "lavis/projects/cap_msvd.yaml")

    # 构造命令行
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=1",
        "--master_port=34654",
        train_py,
        "--cfg-path",
        cfg_path,
        "--options",
        "model.arch", "blip2_vicuna_instruct",
        "model.model_type", "vicuna7b",
        "model.load_finetuned", "False",
        "model.load_pretrained", "False",
        "model.num_query_token", "32",
        "model.vit_precision", "fp16",
        "model.freeze_vit", "True",
        "model.memory_bank_length", "20",
        "model.audio_memory_bank_length", "20",
        "model.num_frames", "40",
        "model.llm_model", llm_path,
        "model.bert_path", bert_path,
        "model.use_audio", "False",
        "model.use_video", "False",
        "model.max_txt_len", "1024",
        "model.max_output_txt_len", "512",
        "model.use_pre_train", "False",
        "model.qwen", "True",
        "run.init_lr", "1e-4",
        "run.max_epoch", "15",
        "run.num_beams", "5",
        "run.batch_size_train", "1",
        "run.batch_size_eval", "1",
        "run.accum_grad_iters", "32",
        "run.num_workers", "6",
        "run.seed", "42",
        "run.evaluate", "False",
        "run.valid_splits", "['val']",
        "run.report_metric", "False",
        "run.prefix", "train",
        "run.job_id", jobid,
    ]

    print("Running command:")
    print(" ".join(cmd))

    result = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)
    if result.returncode != 0:
        sys.exit(result.returncode)

if __name__ == "__main__":
    main()

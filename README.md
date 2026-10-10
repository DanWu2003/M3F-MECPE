# M3F-MECPE

[English](README_EN.md) | 简体中文

本仓库是 ACM MM 2025 论文《Emotion across modalities and cultures: Multilingual multimodal emotion-cause analysis with memory-inspired framework》的官方代码与数据集仓库，包含 MEC<sup>4</sup> 数据集、M<sup>3</sup>F 框架，以及训练和评测代码。

## 论文信息

- **标题**：Emotion across modalities and cultures: Multilingual multimodal emotion-cause analysis with memory-inspired framework
- **作者**：Dan Wu, Xincheng Ju, Dong Zhang, Shoushan Li, Erik Cambria, Guodong Zhou
- **会议**：ACM Multimedia 2025
- **页码**：5775–5783
- **DOI**：[10.1145/3746027.3755655](https://doi.org/10.1145/3746027.3755655)

## MEC<sup>4</sup> 数据集

由于完整数据集体积较大，本仓库不直接存放原始视频等大文件。请通过以下链接下载：

- [MEC⁴ 数据集（Google Drive）](https://drive.google.com/drive/folders/1biCggSJmQOmSA1dn7lGsmNqlFW3y-6Hn?usp=sharing)

下载后的原始数据目录如下：

```text
data/
├── MEC4_text/                 # 文本与标注数据
│   ├── train.json             # 训练集：对话、情感标签及原因配对
│   ├── valid.json             # 验证集
│   └── test.json              # 测试集
├── dialogue_videos/           # 与样本对应的视频片段
│   └── [video_files].mp4
└── subtitles.json             # 每行台词对应的起止时间戳
```

本仓库的 `data/MEC4/` 与 `data/ECF/` 保存了供当前训练代码使用的处理后 JSON 标注。每条样本的主要字段包括：

- `video`：视频或对话标识；
- `frame_length`：提取后的帧数；
- `prompt`：输入模型的对话与任务指令；
- `caption`：目标情绪原因对，例如 `(2,1)|(7,7)`；
- `dia_len`：对话包含的句子数。

模型运行时使用的处理后多模态数据应按以下结构放置：

```text
data/MEC4/
├── cap_train.json
├── cap_val.json
├── cap_test.json
├── video_frames/
│   └── <video_id>/frame000000.jpg
└── audio_mfcc/
    └── <video_id>/audio_000000.npy
```

## M<sup>3</sup>F 框架

M<sup>3</sup>F 面向多语言、多模态情绪原因对抽取（Multimodal Emotion-Cause Pair Extraction, MECPE）。框架在 LAVIS 和 BLIP-2 基础上，将对话文本、视频帧和音频特征映射到大语言模型的输入空间，并生成 `(情绪句编号,原因句编号)` 格式的结果。

```text
对话文本 ─────────────────────────────────────────────┐
视频帧 → 视觉编码器 → Visual Q-Former → 投影层 ───────┤
音频特征 → 音频编码器 → Audio Q-Former → 投影层 ──────┼→ LLM → 情绪原因对
                         时序 Memory Bank ─────────────┘
```

视觉和音频分支均支持时序 Memory Bank。当前实现支持使用 LoRA 或 QLoRA 微调本地大语言模型，并支持在文本基础上选择启用音频和视频模态。

## 仓库结构

```text
M3F/
├── train.py                    # 训练和评测入口
├── run.py                      # Python 启动器
├── run_scripts/                # Slurm、训练和测试脚本
├── data/
│   ├── ECF/                    # ECF 处理后标注
│   ├── MEC4/                   # MEC4 处理后标注
│   └── extract_frames.py       # 视频帧与 MFCC 特征提取
├── LLMs/                       # 本地模型目录，不提交 Git
└── lavis/
    ├── projects/cap_msvd.yaml  # 主实验配置
    ├── models/                 # M³F 与 LAVIS 模型
    ├── datasets/               # 数据集读取与构建
    ├── tasks/                  # MECPE 任务与评测指标
    └── runners/                # 训练循环
```

M<sup>3</sup>F 的主要实现位于：

- `lavis/models/blip2_models/blip2_vicuna_instruct.py`
- `lavis/models/blip2_models/blip2.py`
- `lavis/datasets/datasets/msvd_caption_datasets.py`
- `lavis/datasets/builders/caption_builder.py`
- `lavis/tasks/captioning.py`

由于历史原因，部分类型仍使用 `MSVD` 或 `captioning` 命名；它们在本项目中实际承载 MECPE 数据与任务。

## 环境安装

建议使用 Python 3.9 或更新版本，并根据本机 CUDA 版本安装 PyTorch。随后安装项目依赖：

```bash
pip install -e .
```

大语言模型和 BERT 权重不纳入 Git，默认目录约定为：

```text
LLMs/
├── Qwen3-8B/
└── bert-base-chinese/
```

也可以通过环境变量指定任意本地路径，无需将权重复制到仓库。

## 数据预处理

使用以下命令从原始视频中提取视频帧和对齐的 MFCC 音频特征：

```bash
python data/extract_frames.py \
  --input-dir data/MEC4/dialogue_videos \
  --frame-dir data/MEC4/video_frames \
  --audio-dir data/MEC4/audio_mfcc \
  --fps 10
```

## 训练

训练脚本通过环境变量接收模型路径和实验名称：

```bash
LLM_PATH=LLMs/Qwen3-8B \
BERT_PATH=LLMs/bert-base-chinese \
JOB_ID=train/qwen3 \
bash run_scripts/train.sh
```

也可以运行 Python 启动器：

```bash
LLM_PATH=LLMs/Qwen3-8B \
BERT_PATH=LLMs/bert-base-chinese \
python run.py
```

模型、数据集和优化器参数集中在 `lavis/projects/cap_msvd.yaml` 及启动脚本中。`model.use_audio` 和 `model.use_video` 分别控制音频和视频分支。

## 评测

评测时指定训练得到的 checkpoint：

```bash
CHECKPOINT_PATH=lavis/output/checkpoint_latest.pth \
LLM_PATH=LLMs/Qwen3-8B \
BERT_PATH=LLMs/bert-base-chinese \
JOB_ID=test/qwen3 \
bash run_scripts/test.sh
```

评测代码会从生成文本中提取有效情绪原因对，并报告三组 Precision、Recall 和 F1：

- `pair`：完整情绪原因对；
- `emo`：情绪句识别；
- `cau`：原因句识别。

## 引用

如使用本仓库的代码或 MEC<sup>4</sup> 数据集，请引用：

```bibtex
@inproceedings{wu2025emotion,
  title={Emotion across modalities and cultures: Multilingual multimodal emotion-cause analysis with memory-inspired framework},
  author={Wu, Dan and Ju, Xincheng and Zhang, Dong and Li, Shoushan and Cambria, Erik and Zhou, Guodong},
  booktitle={Proceedings of the 33rd ACM International Conference on Multimedia},
  pages={5775--5783},
  year={2025},
  doi={10.1145/3746027.3755655}
}
```

## 许可证与说明

本项目基于 Salesforce LAVIS 修改。LAVIS 来源文件保留原始版权声明，相关许可证见 `LICENSE`。MEC<sup>4</sup> 数据集、Qwen、BERT 及其他模型或数据资源可能具有各自的使用和分发条款，请在使用前查阅相应许可。

# M3F-MECPE

English | [简体中文](README.md)

This is the official code and dataset repository for the ACM MM 2025 paper “Emotion across modalities and cultures: Multilingual multimodal emotion-cause analysis with memory-inspired framework.” It contains the MEC<sup>4</sup> dataset, the M<sup>3</sup>F framework, and the corresponding training and evaluation code.

## Paper Information

- **Title:** Emotion across modalities and cultures: Multilingual multimodal emotion-cause analysis with memory-inspired framework
- **Authors:** Dan Wu, Xincheng Ju, Dong Zhang, Shoushan Li, Erik Cambria, Guodong Zhou
- **Conference:** ACM Multimedia 2025
- **Pages:** 5775–5783
- **DOI:** [10.1145/3746027.3755655](https://doi.org/10.1145/3746027.3755655)

## MEC<sup>4</sup> Dataset

The complete dataset is too large to be hosted directly in this repository. Download it from:

- [MEC⁴ Dataset on Google Drive](https://drive.google.com/file/d/1rsEo05Y8dqAXrYxotGu9EGBWudMt5U9b/view?usp=sharing)

The downloaded raw dataset has the following layout:

```text
data/
├── MEC4_text/                 # Text and annotation data
│   ├── train.json             # Training dialogues, emotion labels, and cause pairs
│   ├── valid.json             # Validation split
│   └── test.json              # Test split
├── dialogue_videos/           # Video clips associated with the samples
│   └── [video_files].mp4
└── subtitles.json             # Start and end timestamps for each utterance
```

The `data/MEC4/` and `data/ECF/` directories in this repository contain processed JSON annotations consumed by the current training pipeline. The main fields in each sample are:

- `video`: video or dialogue identifier;
- `frame_length`: number of extracted frames;
- `prompt`: dialogue and task instruction supplied to the model;
- `caption`: target emotion-cause pairs, such as `(2,1)|(7,7)`;
- `dia_len`: number of utterances in the dialogue.

Place the processed multimodal inputs in the following layout:

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

## M<sup>3</sup>F Framework

M<sup>3</sup>F addresses multilingual Multimodal Emotion-Cause Pair Extraction (MECPE). Built on LAVIS and BLIP-2, it maps dialogue text, video frames, and audio features into the input space of a large language model and generates results in the form `(emotion utterance index,cause utterance index)`.

```text
Dialogue text ───────────────────────────────────────────────┐
Video frames → Vision encoder → Visual Q-Former → Projector ─┤
Audio features → Audio encoder → Audio Q-Former → Projector ─┼→ LLM → Emotion-cause pairs
                              Temporal Memory Bank ──────────┘
```

Both visual and audio branches support temporal memory banks. The current implementation supports LoRA or QLoRA fine-tuning of a local large language model, with audio and video enabled optionally alongside text.

## Repository Layout

```text
M3F/
├── train.py                    # Training and evaluation entry point
├── run.py                      # Python launcher
├── run_scripts/                # Slurm, training, and evaluation scripts
├── data/
│   ├── ECF/                    # Processed ECF annotations
│   ├── MEC4/                   # Processed MEC4 annotations
│   └── extract_frames.py       # Video-frame and MFCC extraction
├── LLMs/                       # Local models; excluded from Git
└── lavis/
    ├── projects/cap_msvd.yaml  # Main experiment configuration
    ├── models/                 # M³F and LAVIS models
    ├── datasets/               # Dataset loading and construction
    ├── tasks/                  # MECPE task and evaluation metrics
    └── runners/                # Training loops
```

The main M<sup>3</sup>F implementation is located in:

- `lavis/models/blip2_models/blip2_vicuna_instruct.py`
- `lavis/models/blip2_models/blip2.py`
- `lavis/datasets/datasets/msvd_caption_datasets.py`
- `lavis/datasets/builders/caption_builder.py`
- `lavis/tasks/captioning.py`

Some classes retain historical names such as `MSVD` and `captioning`; in this project, they implement the MECPE dataset and task.

## Installation

Python 3.9 or later is recommended. Install PyTorch for the CUDA version available on your system, then install the project dependencies:

```bash
pip install -e .
```

Large-language-model and BERT weights are excluded from Git. The default directory layout is:

```text
LLMs/
├── Qwen3-8B/
└── bert-base-chinese/
```

You may use arbitrary local paths through environment variables instead of copying model weights into the repository.

## Data Preprocessing

Extract video frames and aligned MFCC audio features from the raw videos with:

```bash
python data/extract_frames.py \
  --input-dir data/MEC4/dialogue_videos \
  --frame-dir data/MEC4/video_frames \
  --audio-dir data/MEC4/audio_mfcc \
  --fps 10
```

## Training

The training script accepts model paths and the experiment name through environment variables:

```bash
LLM_PATH=LLMs/Qwen3-8B \
BERT_PATH=LLMs/bert-base-chinese \
JOB_ID=train/qwen3 \
bash run_scripts/train.sh
```

The Python launcher is also available:

```bash
LLM_PATH=LLMs/Qwen3-8B \
BERT_PATH=LLMs/bert-base-chinese \
python run.py
```

Model, dataset, and optimization settings are defined in `lavis/projects/cap_msvd.yaml` and the launcher scripts. `model.use_audio` and `model.use_video` control the audio and video branches, respectively.

## Evaluation

Provide a trained checkpoint when running evaluation:

```bash
CHECKPOINT_PATH=lavis/output/checkpoint_latest.pth \
LLM_PATH=LLMs/Qwen3-8B \
BERT_PATH=LLMs/bert-base-chinese \
JOB_ID=test/qwen3 \
bash run_scripts/test.sh
```

The evaluator extracts valid emotion-cause pairs from generated text and reports precision, recall, and F1 for:

- `pair`: complete emotion-cause pairs;
- `emo`: emotion utterance identification;
- `cau`: cause utterance identification.

## Citation

If you use the code or the MEC<sup>4</sup> dataset, please cite:

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

## License and Acknowledgements

This project is derived from Salesforce LAVIS. Files originating from LAVIS retain their original copyright notices; see `LICENSE` for the applicable BSD 3-Clause License. MEC<sup>4</sup>, Qwen, BERT, and other model or dataset resources may have separate usage and distribution terms. Review their respective licenses before use.

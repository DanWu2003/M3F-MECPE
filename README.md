# M3F-MECPE

本仓库为 ACM MM 2025 论文《Emotion across modalities and cultures: Multilingual multimodal emotion-cause analysis with memory-inspired framework》的官方代码与MEC4数据集仓库。

---

## 论文信息

- **标题**：Emotion across modalities and cultures: Multilingual multimodal emotion-cause analysis with memory-inspired framework  
- **作者**：Dan Wu, Xincheng Ju, Dong Zhang, Shoushan Li, Erik Cambria, Guodong Zhou  
- **会议**：ACM Multimedia 2025  
- **页码**：5775–5783  
- **DOI**：https://doi.org/10.1145/3746027.3755655

---

## $MEC^4$数据集下载

由于数据集体积较大，本仓库不直接存放完整数据文件，请通过外部链接下载：

- 数据集下载链接：[<数据集下载链接>](https://drive.google.com/drive/folders/1biCggSJmQOmSA1dn7lGsmNqlFW3y-6Hn?usp=sharing)

### 数据集目录结构说明
```text
data/
├── MEC4_text/           # 文本与标注数据
│   ├── train.json       # 训练集：包含对话、情感标签及原因配对
│   ├── valid.json       # 验证集
│   └── test.json        # 测试集
├── dialogue_videos/     # 视频资源
│   └── [video_files].mp4 # 与样本对应的视频片段
└── subtitles.json       # 时间轴数据
    └── 包含每行台词在视频中对应的具体时间戳（Start/End Time）
---
```


## 引用方式

如使用本仓库的代码或数据集，请引用以下论文（BibTeX）：

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

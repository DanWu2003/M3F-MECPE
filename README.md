# M3F-MECPE

本仓库为 ACM MM 2025 论文《Emotion across modalities and cultures: Multilingual multimodal emotion-cause analysis with memory-inspired framework》的官方代码与资源仓库。

---

## 论文信息

**论文标题**：Emotion across modalities and cultures: Multilingual multimodal emotion-cause analysis with memory-inspired framework  
**作者**：Dan Wu, Xincheng Ju, Dong Zhang, Shoushan Li, Erik Cambria, Guodong Zhou  
**会议**：Proceedings of the 33rd ACM International Conference on Multimedia (ACM MM 2025)  
**页码**：5775–5783  
**DOI（正式出版版本）**：https://doi.org/10.1145/3746027.3755655

---

## 数据集下载

由于数据集体积较大，本仓库不直接存放完整数据文件，请通过外部链接下载：

- 数据集下载链接：<数据集下载链接>
- （可选）备用链接：<备用下载链接>

下载并解压后，建议按以下目录结构放置（如结构不同，请相应修改运行命令中的路径参数）：

```bash
data/
  train.json
  dev.json
  test.json
````

---

## 环境安装

推荐 Python 3.9+。

在仓库根目录执行：

```bash
pip install -r requirements.txt
```

---

## 代码运行

> 说明：以下为通用运行示例，请根据仓库中实际脚本与参数名调整（通常位于 `scripts/` 目录）。

### 1) 推理 / 生成预测结果

```bash
python scripts/infer.py \
  --data_path data/test.json \
  --output_path outputs/pred.json
```

### 2) 评测

```bash
python scripts/eval.py \
  --pred_path outputs/pred.json \
  --gold_path data/test.json
```

---

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

"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import os
import re
import logging
from lavis.common.dist_utils import main_process
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask
import collections
import pdb

@registry.register_task("captioning")
class CaptionTask(BaseTask):
    def __init__(self, num_beams, max_len, min_len, evaluate, report_metric=True, dataset_name=None):
        super().__init__()

        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len
        self.evaluate = evaluate

        self.report_metric = report_metric
        self.dataset_name = dataset_name

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        num_beams = run_cfg.num_beams
        max_len = run_cfg.max_len
        min_len = run_cfg.min_len
        evaluate = run_cfg.evaluate

        report_metric = run_cfg.get("report_metric", True)

        return cls(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            evaluate=evaluate,
            report_metric=report_metric,
            dataset_name=list(cfg.datasets_cfg.keys())[0],
        )

    def valid_step(self, model, samples):
        results = []

        # run_cfg = slf.cfg.run_cfg
        captions = model.generate(
            samples,
            use_nucleus_sampling=False,
            num_beams=self.num_beams,
            max_length=self.max_len,
            min_length=self.min_len,
        )

        img_ids = samples["image_id"]
        # Ids = samples["id"] Ids.tolist()
        for caption, img_id in zip(captions, img_ids):
            if 'coco' in self.dataset_name:
                results.append({"caption": caption, "image_id": int(img_id)})
            else:
                results.append({"caption": caption, "image_id": img_id})

        return results

    def after_evaluation(self, val_result, split_name, epoch, dataset, **kwargs):
        eval_result_file = self.save_result(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_epoch{}".format(split_name, epoch),
            remove_duplicate="image_id",
        )

        if self.report_metric:
            if 'coco' in self.dataset_name:
                metrics = self._report_metrics_coco(
                    eval_result_file=eval_result_file, split_name=split_name
                )
            else:
                metrics = self._report_metrics_video_caption(
                    eval_result_file=eval_result_file, split_name=split_name, dataset=dataset, dataset_name=self.dataset_name
                )
        else:
            label_file = os.path.join(
                os.path.dirname(dataset.audio_root), f"cap_{split_name}.json"
            )
            in_file = os.path.join(os.path.dirname(eval_result_file), "generate_" + os.path.basename(eval_result_file))
            evaluator = EmotionCauseEvaluator(
                label_file=label_file,
                generate_file=eval_result_file,
                in_file=in_file
            )

            evaluator.aggregate()
            prf = evaluator.cal_prf()
            logging.info(f"PRF results: {prf}")
            pair_f = prf['pair']['F1']
            metrics = {"agg_metrics": pair_f}

        return metrics

    @main_process
    def _report_metrics_coco(self, eval_result_file, split_name):

        # TODO better way to define this
        coco_gt_root = os.path.join(registry.get_path("cache_root"), "coco_gt")
        coco_val = coco_caption_eval(coco_gt_root, eval_result_file, split_name)

        agg_metrics = coco_val.eval["CIDEr"] + coco_val.eval["Bleu_4"]
        log_stats = {split_name: {k: v for k, v in coco_val.eval.items()}}

        with open(
            os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")

        coco_res = {k: v for k, v in coco_val.eval.items()}
        coco_res["agg_metrics"] = agg_metrics

        return coco_res

    @main_process
    def _report_metrics_video_caption(self, eval_result_file, split_name, dataset, dataset_name):
        filenames = {
            "train": "cap_train.json",
            "val": "cap_val.json",
            "test": "cap_test.json",
        }
        if 'msvd' in dataset_name:
            annotation_file = os.path.join(
                registry.get_path("cache_root"),
                "msvd/annotation/coco_gt",
                filenames[split_name],
            )
        elif 'msrvtt' in dataset_name:
            annotation_file = os.path.join(registry.get_path("cache_root"), "msrvtt/annotation/coco_gt", filenames[split_name])
        elif 'youcook2' in dataset_name:
            annotation_file = os.path.join(registry.get_path("cache_root"), "youcook2/annotation/coco_gt", filenames[split_name])
        elif 'jyen' in dataset_name:
            annotation_file = os.path.join(registry.get_path("cache_root"), "jyen/annotation/coco_gt", filenames[split_name])

        print('annotation_file: ', annotation_file)
        # create coco object and coco_result object
        coco = COCO(annotation_file)
        coco_result = coco.loadRes(eval_result_file)

        # create coco_eval object by taking coco and coco_result
        coco_eval = COCOEvalCap(coco, coco_result)

        # evaluate results
        # SPICE will take a few minutes the first time, but speeds up due to caching
        coco_eval.evaluate()

        # print output evaluation scores
        for metric, score in coco_eval.eval.items():
            print(f"{metric}: {score:.3f}")

        agg_metrics = coco_eval.eval["CIDEr"] + coco_eval.eval["Bleu_4"] + coco_eval.eval["METEOR"] + coco_eval.eval["ROUGE_L"]
        log_stats = {split_name: {k: v for k, v in coco_eval.eval.items()}}

        coco_res = {k: v for k, v in coco_eval.eval.items()}
        coco_res["agg_metrics"] = agg_metrics
        return coco_res


# TODO better structure for this.
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO
from torchvision.datasets.utils import download_url


def coco_caption_eval(coco_gt_root, results_file, split):
    urls = {
        "val": "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val_gt.json",
        "test": "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test_gt.json",
    }
    filenames = {
        "val": "coco_karpathy_val_gt.json",
        "test": "coco_karpathy_test_gt.json",
    }

    download_url(urls[split], coco_gt_root)
    annotation_file = os.path.join(coco_gt_root, filenames[split])

    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    # coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f"{metric}: {score:.3f}")

    return coco_eval


class EmotionCauseEvaluator:
    def __init__(self, label_file, generate_file, in_file):
        self.label_file = label_file
        self.generate_file = generate_file
        self.in_file = in_file

    def my_filter(self, pairs, dia_len):
        """过滤无效的情感和原因对"""
        filtered = set()
        target = r'\d+,\d+'
        res = re.findall(target, pairs)
        for pair in res:
            emo, cau = pair.split(',')[0], pair.split(',')[-1]
            if int(emo) > dia_len or int(cau) > dia_len:
                continue  # 忽略不符合条件的情感原因对
            else:
                filtered.add(pair)
        return filtered

    def filter_label(self, pairs):
        """过滤标签中的括号，并将其分割成列表"""
        if not pairs.strip():
            return []
        return pairs.replace('(', '').replace(')', '').split('|')

    def aggregate(self):
        """聚合标签和生成结果"""
        all_aggregated = {}
        with open(self.label_file, 'r') as f:
            label_data = json.load(f)
        with open(self.generate_file, 'r') as f1:
            generate_data = json.load(f1)

        generated_by_video = {
            str(item['image_id']): item.get('caption', '') for item in generate_data
        }

        for k, v in label_data.items():
            video = v['video']
            label = self.filter_label(v['caption'])
            generated = generated_by_video.get(str(video), '')
            dia_len = v["dia_len"]
            generated = self.my_filter(generated, dia_len)
            all_aggregated[video] = []
            all_aggregated[video].append(label)
            all_aggregated[video].append(list(generated))

        with open(self.in_file, 'w') as f2:
            json.dump(all_aggregated, f2, indent=4, ensure_ascii=False)

    def cal_prf(self):
        """计算精确度、召回率和F1分数"""
        pred_correct, pred_positive, positive_samples = 0, 0, 0
        pred_correct_emo, pred_positive_emo, positive_samples_emo = 0, 0, 0
        pred_correct_cau, pred_positive_cau, positive_samples_cau = 0, 0, 0
        with open(self.in_file, 'r') as f:
            data = json.load(f)

        for k, v in data.items():
            label = v[0]
            generated = v[1]
            pred_positive += len(generated)
            positive_samples += len(label)
            for p in generated:
                if p in label:
                    pred_correct += 1
            emo_utts = set([s.split(',')[0] for s in label])
            pred_emo_utts = set([s.split(',')[0] for s in generated])
            cau_utts = set([s.split(',')[1] for s in label])
            pred_cau_utts = set([s.split(',')[1] for s in generated])
            pred_positive_emo += len(pred_emo_utts)
            positive_samples_emo += len(emo_utts)
            for u in pred_emo_utts:
                if u in emo_utts:
                    pred_correct_emo += 1
            pred_positive_cau += len(pred_cau_utts)
            positive_samples_cau += len(cau_utts)
            for u in pred_cau_utts:
                if u in cau_utts:
                    pred_correct_cau += 1

        # 计算Pair、Emotion、Cause的精确度、召回率和F1
        pair_p = pred_correct / pred_positive if pred_positive else 0.0
        pair_r = pred_correct / positive_samples if positive_samples else 0.0
        pair_f = 2 * pair_p * pair_r / (pair_p + pair_r + 1e-8)

        emo_p = pred_correct_emo / pred_positive_emo if pred_positive_emo else 0.0
        emo_r = pred_correct_emo / positive_samples_emo if positive_samples_emo else 0.0
        emo_f = 2 * emo_p * emo_r / (emo_p + emo_r + 1e-8)

        cau_p = pred_correct_cau / pred_positive_cau if pred_positive_cau else 0.0
        cau_r = pred_correct_cau / positive_samples_cau if positive_samples_cau else 0.0
        cau_f = 2 * cau_p * cau_r / (cau_p + cau_r + 1e-8)

        return {
            'pair': {'P': round(pair_p, 5), 'R': round(pair_r, 5), 'F1': round(pair_f, 5)},
            'emo': {'P': round(emo_p, 5), 'R': round(emo_r, 5), 'F1': round(emo_f, 5)},
            'cau': {'P': round(cau_p, 5), 'R': round(cau_r, 5), 'F1': round(cau_f, 5)}
        }

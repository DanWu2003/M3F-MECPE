"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json
import re
from PIL import Image

import pandas as pd
import numpy as np
import torch
from torchvision.transforms.functional import pil_to_tensor

from lavis.datasets.datasets.video_caption_datasets import VideoCaptionDataset
from lavis.datasets.data_utils import load_video
import pdb


class MSVDCapDataset(VideoCaptionDataset):
    def __init__(
        self,
        vis_processor,
        audio_processor,
        text_processor,
        vis_root,
        audio_root,
        ann_paths,
        num_frames,
        prompt="",
        split="train",
        use_audio=True,
        use_video=True,
    ):
        self.vis_root = vis_root
        self.audio_root = audio_root

        self.annotation = {}
        for ann_path in ann_paths:
            self.annotation.update(json.load(open(ann_path)))
        self.video_id_list = list(self.annotation.keys())
        self.video_id_list.sort()
        self.fps = 10

        self.num_frames = num_frames
        self.vis_processor = vis_processor
        self.audio_processor = audio_processor
        self.text_processor = text_processor
        self.prompt = prompt
        self.use_audio = use_audio
        self.use_video = use_video

    def __getitem__(self, index):
        video_id = self.video_id_list[index]
        ann = self.annotation[video_id]
        # Divide the range into num_frames segments and select a random index from each segment
        # change

        segment_list = np.linspace(0, ann['frame_length'], self.num_frames + 1, dtype=int)
        segment_start_list = segment_list[:-1]
        segment_end_list = segment_list[1:]
        selected_frame_index = []
        for start, end in zip(segment_start_list, segment_end_list):
            if start == end:
                selected_frame_index.append(start)
            else:
                selected_frame_index.append(np.random.randint(start, end))

        # change
        frame_list = []
        audio_frame_list = []
        for i, frame_index in enumerate(selected_frame_index):
            if self.use_video:
                frame_path = os.path.join(
                    self.vis_root, ann["video"], f"frame{frame_index:06d}.jpg"
                )
                frame = Image.open(frame_path).convert("RGB")
                frame_list.append(pil_to_tensor(frame).to(torch.float32))

            if self.use_audio:
                audio_file_path = os.path.join(
                    self.audio_root, ann["video"], f"audio_{frame_index:06d}.npy"
                )
                audio_frame = np.load(audio_file_path, allow_pickle=False)
                if audio_frame.shape[1] > 9:
                    audio_frame = audio_frame[:, :9]
                audio_frame = np.pad(
                    audio_frame,
                    ((0, 0), (0, 9 - audio_frame.shape[1])),
                    mode="constant",
                )
                audio_frame_list.append(torch.tensor(audio_frame, dtype=torch.float32))

        video = torch.empty(0)
        if self.use_video:
            video = self.vis_processor(torch.stack(frame_list, dim=1))

        audio = torch.empty(0)
        if self.use_audio:
            audio = self.audio_processor(torch.stack(audio_frame_list, dim=1))

        text_input = ann['prompt']
        caption = self.text_processor.pre_caption(ann["caption"])


        return {
            "image": video,
            "audio": audio,
            "text_input": text_input,
            "text_output": caption,
            "prompt": text_input,
            "image_id": ann["video"],
            "id":ann["id"],
        }

    def __len__(self):
        return len(self.video_id_list)

class MSVDCapEvalDataset(MSVDCapDataset):
    def __init__(
        self,
        vis_processor,
        audio_processor,
        text_processor,
        vis_root,
        audio_root,
        ann_paths,
        num_frames,
        prompt,
        split="val",
        use_audio=True,
        use_video=True,
    ):
        super().__init__(
            vis_processor,
            audio_processor,
            text_processor,
            vis_root,
            audio_root,
            ann_paths,
            num_frames,
            prompt,
            split=split,
            use_audio=use_audio,
            use_video=use_video,
        )

    def __getitem__(self, index):
        video_id = self.video_id_list[index]
        ann = self.annotation[video_id]

        selected_frame_index = np.rint(np.linspace(0, ann['frame_length']-1, self.num_frames)).astype(int).tolist()
        frame_list = []
        audio_frame_list = []
        for i, frame_index in enumerate(selected_frame_index):
            if self.use_video:
                frame_path = os.path.join(
                    self.vis_root, ann["video"], f"frame{frame_index:06d}.jpg"
                )
                frame = Image.open(frame_path).convert("RGB")
                frame_list.append(pil_to_tensor(frame).to(torch.float32))

            if self.use_audio:
                audio_file_path = os.path.join(
                    self.audio_root, ann["video"], f"audio_{frame_index:06d}.npy"
                )
                audio_frame = np.load(audio_file_path, allow_pickle=False)
                if audio_frame.shape[1] > 9:
                    audio_frame = audio_frame[:, :9]
                audio_frame = np.pad(
                    audio_frame,
                    ((0, 0), (0, 9 - audio_frame.shape[1])),
                    mode="constant",
                )
                audio_frame_list.append(torch.tensor(audio_frame, dtype=torch.float32))

        video = torch.empty(0)
        if self.use_video:
            video = self.vis_processor(torch.stack(frame_list, dim=1))

        audio = torch.empty(0)
        if self.use_audio:
            audio = self.audio_processor(torch.stack(audio_frame_list, dim=1))

        text_input = ann['prompt']
        caption = self.text_processor.pre_caption(ann["caption"])

        return {
            "image": video,
            "audio": audio,
            "text_input": text_input,
            "text_output": caption,
            "prompt": text_input,
            "image_id": ann["video"],
            "id":ann["id"],
        }

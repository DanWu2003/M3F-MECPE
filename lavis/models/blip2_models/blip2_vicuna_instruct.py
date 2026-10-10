"""
Requires Transformer 4.28 and above, implementation may change according the Llama implementation
"""
import logging
import string
from packaging import version

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

import transformers

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train, memory_bank_compress
import copy
import numpy as np
import sys

@registry.register_model("blip2_vicuna_instruct_m3f")
class Blip2VicunaInstructM3F(Blip2Base):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "vicuna7b": "configs/models/blip2/blip2_instruct_vicuna7b.yaml",
        "vicuna13b": "configs/models/blip2/blip2_instruct_vicuna13b.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        llm_model="",
        prompt="",
        max_txt_len=256,  # change 128 512
        max_output_txt_len=32, # change 256 32
        apply_lemmatizer=False,
        qformer_text_input=True,
        memory_bank_length=0,
        audio_memory_bank_length=0,
        num_frames=0,
        use_audio=True,
        use_video=True,
        audio_input_size=9,
        audio_output_size=128,
        use_LoRA=True,
        use_QLoRA=True,
        bert_path=None,
        qwen=False,
    ):
        super().__init__()
        transformers_version = version.parse(transformers.__version__)
        assert transformers_version >= version.parse("4.28"), "BLIP-2 Vicuna requires transformers>=4.28"
        from transformers import LlamaTokenizer, AutoTokenizer, LlamaForCausalLM, AutoModelForCausalLM
        # from lavis.models.blip2_models.modeling_llama import LlamaForCausalLM

        self.tokenizer = self.init_tokenizer(truncation_side="left", bert_path=bert_path)  # bert-base-chinese

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )

        # change
        self.audio_encoder, self.ln_audio = self.init_audio_encoder(
            audio_input_size, audio_output_size
        )

        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, memory_bank_length=memory_bank_length, num_frames=num_frames, bert_path=bert_path
        )
        if not qformer_text_input:
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
        else:
            self.Qformer.resize_token_embeddings(len(self.tokenizer))
        self.Qformer.cls = None

        #     音频部分  !!!!!!
        self.audio_Qformer, self.audio_query_tokens = self.init_Qformer(
            num_query_token, audio_output_size, memory_bank_length=audio_memory_bank_length, num_frames=num_frames, bert_path=bert_path
        )
        if not qformer_text_input:
            self.audio_Qformer.bert.embeddings.word_embeddings = None
            self.audio_Qformer.bert.embeddings.position_embeddings = None
            for layer in self.audio_Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
        else:
            self.audio_Qformer.resize_token_embeddings(len(self.tokenizer))
        self.audio_Qformer.cls = None
        ### 音频部分Qformer end

        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model, use_fast=False, truncation_side="left")
        # change QLoRA or LoRA
        if use_LoRA or use_QLoRA:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            from transformers import BitsAndBytesConfig
            import bitsandbytes as bnb
            from peft.tuners.lora import LoraLayer

        if use_QLoRA:
        # 使用 Q-LoRA 加载模型
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype='float32',
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            # quantization_config = BitsAndBytesConfig(
            #     load_in_4bit=True,  # 启用 8-bit 量化
            #     bnb_8bit_compute_dtype='float32',  # 计算时使用的浮动精度，通常为 float32
            #     bnb_8bit_quant_type="fp8",  # 量化类型：可以是 fp8（通常使用），也可以根据需求调整
            #     bnb_8bit_use_double_quant=True,  # 是否启用双重量化（针对更高效的量化过程）
            # )
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                llm_model,
                torch_dtype=torch.float32,
                # load_in_4bit=True,
                quantization_config=quantization_config,
            )
        else:
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                llm_model,
                torch_dtype=torch.float32,
            )
        # LoRA模块
        def find_all_linear_names(model):
            # cls = bnb.nn.Linear4bit if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
            if use_QLoRA:
                cls = bnb.nn.Linear4bit
            else:
                cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if isinstance(module, cls):
                    names = name.split('.')
                    lora_module_names.add(names[0] if len(names) == 1 else names[-1])


            if 'lm_head' in lora_module_names: # needed for 16-bit
                lora_module_names.remove('lm_head')
            return list(lora_module_names)
        if use_LoRA:
        # self.llm_model = prepare_model_for_kbit_training(self.llm_model, use_gradient_checkpointing=False)
            modules = find_all_linear_names(self.llm_model)
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules=modules, # "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.llm_model = get_peft_model(self.llm_model, lora_config)

            # 稳定模型模块, 保证训练稳定性
            for name, module in self.llm_model.named_modules():
                if isinstance(module, LoraLayer):
                    module = module.to(torch.float32)
                if 'norm' in name:
                    module = module.to(torch.float32)
                if 'lm_head' in name or 'embed_tokens' in name:
                    if hasattr(module, 'weight'):
                        if module.weight.dtype == torch.float32:
                            module = module.to(torch.float32)

            self.llm_model.print_trainable_parameters()
        # self.llm_model = LlamaForCausalLM.from_pretrained(
        #     llm_model, torch_dtype=torch.float16
        # )
        # self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
        # self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        # self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})
        # self.llm_tokenizer.pad_token = self.llm_tokenizer.unk_token

        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

        # self.eos_token_id = self.llm_tokenizer(
        #     self.llm_tokenizer.eos_token, add_special_tokens=False
        # ).input_ids[0]

        # change
        # print(self.llm_model.lm_head.weight.dtype)
        # self.llm_model.lm_head = self.llm_model.lm_head.float()  # change
        # for name, param in self.llm_model.named_parameters():
        #     if name == 'lm_head.weight':
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False

        self.llm_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llm_model.config.hidden_size
        )

        self.llm_proj_audio = nn.Linear(
            self.audio_Qformer.config.hidden_size, self.llm_model.config.hidden_size
        )

        self.max_txt_len = max_txt_len
        print('max_txt_len: ', max_txt_len)
        self.max_output_txt_len = max_output_txt_len
        print('max_output_txt_len: ', max_output_txt_len)
        self.prompt = prompt
        # print('prompt: ', prompt)
        prompt_tokens = self.llm_tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.attention_mask.sum(1)

        self._lemmatizer = None

        self.qformer_text_input = qformer_text_input
        self.num_query_token = num_query_token
        self.memory_bank_length = memory_bank_length
        self.audio_memory_bank_length = audio_memory_bank_length
        self.use_memory_bank = True if memory_bank_length > 0 else False
        self.use_memory_bank_audio = True if audio_memory_bank_length > 0 else False
        self.visual_memory_bank = None

        # change
        self.audio_memory_bank = None

        self.image_pe = nn.Embedding(120, 1408)
        nn.init.constant_(self.image_pe.weight, 0.0)
        # change
        self.audio_pe = nn.Embedding(120, audio_output_size)
        # self.audio_pe = nn.Embedding(120, 1408)
        nn.init.constant_(self.audio_pe.weight, 0.0)
        self.use_audio=use_audio
        self.use_video=use_video
        self.qwen=qwen
        # self.learnable_token_embeddings = nn.Parameter(torch.randn(5, self.llm_model.config.hidden_size))
        # self.audio_weights = nn.Parameter(torch.randn(40))
        # self.weights = nn.Parameter(torch.randn(40))

        # self.temporal_attn = nn.MultiheadAttention(embed_dim=1408, num_heads=8, batch_first=True)
        # self.temporal_attn_audio = nn.MultiheadAttention(embed_dim=audio_output_size, num_heads=8, batch_first=True)

    def concat_text_input_output(self, input_ids, input_atts, output_ids, output_atts):
        input_part_targets_len = []
        llm_tokens = {"input_ids": [], "attention_mask": []}
        for i in range(input_ids.size(0)):
            this_input_ones = input_atts[i].sum()
            input_part_targets_len.append(this_input_ones)
            llm_tokens['input_ids'].append(
                torch.cat([
                    input_ids[i][:this_input_ones],
                    output_ids[i][1:],
                    input_ids[i][this_input_ones:]
                ])
            )
            llm_tokens['attention_mask'].append(
                torch.cat([
                    input_atts[i][:this_input_ones],
                    output_atts[i][1:],
                    input_atts[i][this_input_ones:]
                ])
            )
        llm_tokens['input_ids'] = torch.stack(llm_tokens['input_ids'])
        llm_tokens['attention_mask'] = torch.stack(llm_tokens['attention_mask'])
        return llm_tokens, input_part_targets_len

    def forward(self, samples):
        image = samples["image"]
        audio = samples["audio"]
        # print("Bert Input", samples["text_input"])
        if self.use_audio or self.use_video:
            text_Qformer = self.tokenizer(
                samples["text_input"],  # [B]
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            )
            text_Qformer_audio = copy.deepcopy(text_Qformer)
            text_Qformer = {key: value.to(image.device) if isinstance(value, torch.Tensor) else value for key, value in text_Qformer.items()}
            text_Qformer_audio = {key: value.to(audio.device) if isinstance(value, torch.Tensor) else value for key, value in text_Qformer_audio.items()}

        if self.use_video:
            if image.dim() == 5:
                B, C, T, H, W = image.shape

            if self.qformer_text_input:
                # if is_video:
                #     text_input = [text for text in samples["text_input"] for _ in range(T)]
                # else:
                #     text_input = samples["text_input"]
                # 不用mb 修改下面的else，使音频和视频共享一个text_Qformer提个速度，要不然要tokenizer两次
                if self.use_memory_bank:
                    query_tokens = self.query_tokens.expand(B, -1, -1) # [B, 32, C]
                    query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device) # [B, N]
                    Qformer_atts = torch.cat([query_atts, text_Qformer["attention_mask"]],dim=1)
                    for t in range(T):
                        with self.maybe_autocast():
                            image_embeds = self.ln_vision(self.visual_encoder(image[:, :, t, :, :])) #[B, 256+1, 1408]
                        N, C = image_embeds.shape[-2:]
                        position_ids = torch.tensor([t]).long().to(image_embeds.device) #[1]
                        position_ids = position_ids.unsqueeze(0).expand(B, -1) #[B, 1]
                        image_embeds = image_embeds + self.image_pe(position_ids) # [B, N, C]
                        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device) # [B, N]
                        image_embeds = image_embeds.unsqueeze(1) # [B, 1, N, C]

                        if t == 0:
                            encoder_hidden_states = image_embeds # [B, 1, N, C]
                            self.size_constant = torch.ones(B, 1, N).to(image_embeds.device) # [B, 1, N]
                        else:
                            encoder_hidden_states = torch.cat([self.visual_memory_bank, image_embeds], dim=1) # [B, (t+1), N, C]

                        query_output = self.Qformer.bert(
                            text_Qformer["input_ids"],
                            attention_mask=Qformer_atts,
                            query_embeds=query_tokens,
                            encoder_hidden_states=encoder_hidden_states.view(B, -1, C),
                            encoder_attention_mask=image_atts,
                            return_dict=True,
                        )

                        # If it is the first frame, initialize the visual_memory_bank as the embedding of the first frame
                        # If not, concatenate the visual_memory_bank with the current frame embedding and update the compression_size
                        if t == 0:
                            self.visual_memory_bank = image_embeds.detach()  # [B, 1, N, C]
                            self.compression_size = self.size_constant  # [B, 1, N]
                        else:
                            self.visual_memory_bank = torch.cat([self.visual_memory_bank, image_embeds.detach()], dim=1)  # [B, t+1, N, C]
                            self.compression_size = torch.cat([self.compression_size, self.size_constant], dim=1)  # [B, t+1, N]

                        # If it is the last frame, delete the visual_memory_bank and compression_size
                        # Else, if the current length of the visual_memory_bank exceeds the threshold, compress the visual_memory_bank
                        if t == T - 1:
                            del self.visual_memory_bank
                            del self.compression_size
                        elif self.visual_memory_bank.size(1) > self.memory_bank_length:
                            self.visual_memory_bank, self.compression_size = memory_bank_compress(self.visual_memory_bank, self.compression_size)
                else:
                    # logging.info("=====Don't Use Memory Bank Video=====")
                    query_tokens = self.query_tokens.expand(B, -1, -1)
                    query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
                    Qformer_atts = torch.cat([query_atts, text_Qformer["attention_mask"]], dim=1)

                    # 方法一
                    image_embeds_avg = None
                    image_atts_avg = None

                    for t in range(T):
                        with self.maybe_autocast():
                            image_embeds = self.ln_vision(self.visual_encoder(image[:, :, t, :, :]))  # [B, 256+1, 1408]
                        N, C = image_embeds.shape[-2:]

                        position_ids = torch.tensor([t]).long().to(image_embeds.device)  # [1]
                        position_ids = position_ids.unsqueeze(0).expand(B, -1)  # [B, 1]
                        image_embeds = image_embeds + self.image_pe(position_ids)  # [B, N, C]

                        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)  # [B, N]

                        if image_embeds_avg is None:
                            image_embeds_avg = image_embeds.unsqueeze(1) # [B, 1, N, C]
                            image_atts_avg = image_atts.unsqueeze(1)  # [B, 1, N]
                        else:
                            image_embeds_avg += image_embeds.unsqueeze(1)  # [B, T, N, C]
                            image_atts_avg += image_atts.unsqueeze(1) # [B, T, N]

                    # Now average the embeddings and attention masks across time steps (T frames)
                    image_atts_avg = image_atts_avg.float()
                    image_embeds_avg /= T  # [B, T, N, C]
                    image_atts_avg /= T # [B, T, N]
                    image_atts_avg = image_atts_avg.long()

                    # 方法二
                    # image_embeds_avg = None
                    # image_atts_avg = None

                    # soft_weights = torch.softmax(self.weights, dim=0).to(image.device)

                    # for t in range(T):
                    #     with self.maybe_autocast():
                    #         image_embeds = self.ln_vision(self.visual_encoder(image[:, :, t, :, :]))  # [B, 256+1, 1408]
                    #     N, C = image_embeds.shape[-2:]

                    #     position_ids = torch.tensor([t]).long().to(image_embeds.device)  # [1]
                    #     position_ids = position_ids.unsqueeze(0).expand(B, -1)  # [B, 1]
                    #     image_embeds = image_embeds + self.image_pe(position_ids)  # [B, N, C]

                    #     image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)  # [B, N]

                    #     if image_embeds_avg is None:
                    #         image_embeds_avg = image_embeds.unsqueeze(1)*soft_weights[t]  # [B, 1, N, C]
                    #         image_atts_avg = image_atts.unsqueeze(1)*soft_weights[t]  # [B, 1, N]
                    #     else:
                    #         image_embeds_avg += image_embeds.unsqueeze(1)*soft_weights[t]  # [B, T, N, C]
                    #         image_atts_avg += image_atts.unsqueeze(1)*soft_weights[t]  # [B, T, N]

                    # # Now average the embeddings and attention masks across time steps (T frames)
                    # image_atts_avg = image_atts_avg.float()
                    # image_embeds_avg /= soft_weights.sum()  # [B, T, N, C]
                    # image_atts_avg /= soft_weights.sum()  # [B, T, N]
                    # image_atts_avg = image_atts_avg.long()

                    # 方法三
                    # image_embeds_list = []  # 保存每帧的图像特征，后续拼接
                    # image_atts_list = []

                    # for t in range(T):
                    #     with self.maybe_autocast():
                    #         image_embeds = self.ln_vision(self.visual_encoder(image[:, :, t, :, :]))  # [B, N, C]

                    #     N, C = image_embeds.shape[-2:]

                    #     # 添加位置编码
                    #     position_ids = torch.tensor([t], device=image.device).unsqueeze(0).expand(B, -1)
                    #     image_embeds = image_embeds + self.image_pe(position_ids)

                    #     image_embeds_list.append(image_embeds)  # [B, N, C]
                    #     image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)  # [B, N]
                    #     image_atts_list.append(image_atts)

                    # # 将 T 帧堆叠起来: [B, T, N, C]
                    # image_embeds_stack = torch.stack(image_embeds_list, dim=1)  # [B, T, N, C]

                    # # 准备输入 MHA：合并 B 和 N，为了对每个 patch 跨 T 融合
                    # image_embeds_mha = image_embeds_stack.permute(0, 2, 1, 3).reshape(B * N, T, C)  # [B*N, T, C]

                    # # 构建 MHA 模块（或在 __init__ 中定义）
                    # self.temporal_attn.to(image.device)

                    # # 使用 self-attention 融合时间信息（query=key=value）
                    # attn_out, _ = self.temporal_attn(image_embeds_mha, image_embeds_mha, image_embeds_mha)  # [B*N, T, C]

                    # # 聚合时间维度（平均或取第一个也可以）
                    # attn_out = attn_out.mean(dim=1)  # [B*N, C]

                    # # 恢复到 [B, N, C]
                    # image_embeds_avg = attn_out.view(B, N, C).unsqueeze(1)

                    # image_atts_stack = torch.stack(image_atts_list, dim=1)
                    # image_atts_avg = image_atts_stack.float().mean(dim=1, keepdim=True)
                    # image_atts_avg = (image_atts_avg > 0.5).long()

                    # Now we can use image_embeds_avg and image_atts_avg as inputs to the model
                    # Continue with the query_output generation
                    query_output = self.Qformer.bert(
                        text_Qformer["input_ids"],
                        attention_mask=Qformer_atts,
                        query_embeds=query_tokens,
                        encoder_hidden_states=image_embeds_avg.squeeze(1),  # Flatten for the batch dimension [B*T, N, C]
                        encoder_attention_mask=image_atts_avg.squeeze(1),  # Flatten for the batch dimension [B*T, N]
                        return_dict=True,
                    )
            else:
                query_tokens = self.query_tokens.expand(B * T, -1, -1)
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
            inputs_llm = self.llm_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
            # inputs_llm_np = inputs_llm.detach().cpu().numpy()
            if self.use_video:
                inputs_llm = inputs_llm.reshape(B, -1, inputs_llm.shape[-1])
            atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image.device)

        # 音频部分!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if self.use_audio:
            if audio.dim() == 4:
                B, F, T, C = audio.shape  # B: batch size, F: feature channels, T: time frames (audio frames)

            if self.qformer_text_input:
                # if is_audio:
                #     text_input_audio = [text for text in samples["text_input"] for _ in range(T)]
                # else:
                #     text_input_audio = samples["text_input"]
                if self.use_memory_bank_audio:
                    audio_query_tokens = self.audio_query_tokens.expand(B, -1, -1)  # [B, 32, C]
                    query_atts_audio = torch.ones(audio_query_tokens.size()[:-1], dtype=torch.long).to(audio.device)  # [B, N]
                    Qformer_atts_audio = torch.cat([query_atts_audio, text_Qformer_audio["attention_mask"]], dim=1)

                    for t in range(T):
                        with self.maybe_autocast():
                            audio_embeds = self.ln_audio(self.audio_encoder(audio[:, :, t, :]))  # [B, 128]
                        N, C = audio_embeds.shape[-2:]
                        position_ids_audio = torch.tensor([t]).long().to(audio_embeds.device)  # [1]
                        position_ids_audio = position_ids_audio.unsqueeze(0).expand(B, -1)  # [B, 1]
                        audio_embeds = audio_embeds + self.audio_pe(position_ids_audio)  # [B, N]
                        audio_atts = torch.ones(audio_embeds.size()[:-1], dtype=torch.long).to(audio.device)  # [B, N]
                        audio_embeds = audio_embeds.unsqueeze(1)  # [B, 1, N, C]

                        if t == 0:
                            encoder_hidden_states_audio = audio_embeds  # [B, 1, N, C]
                            self.size_constant_audio = torch.ones(B, 1, N).to(audio_embeds.device)  # [B, 1, N]
                        else:
                            encoder_hidden_states_audio = torch.cat([self.audio_memory_bank, audio_embeds], dim=1)  # [B, (t+1), N, C]

                        query_output_audio = self.audio_Qformer.bert(
                            text_Qformer_audio["input_ids"],
                            attention_mask=Qformer_atts_audio,
                            query_embeds=audio_query_tokens,
                            encoder_hidden_states=encoder_hidden_states_audio.view(B, -1, C),
                            encoder_attention_mask=audio_atts,
                            return_dict=True,
                        )

                        # If it is the first frame, initialize the audio_memory_bank as the embedding of the first frame
                        if t == 0:
                            self.audio_memory_bank = audio_embeds.detach()  # [B, 1, N, C]
                            self.audio_compression_size = self.size_constant_audio  # [B, 1, N]
                        else:
                            self.audio_memory_bank = torch.cat([self.audio_memory_bank, audio_embeds.detach()], dim=1)  # [B, t+1, N, C]
                            self.audio_compression_size = torch.cat([self.audio_compression_size, self.size_constant_audio], dim=1)  # [B, t+1, N]

                        # If it is the last frame, delete the audio_memory_bank and audio_compression_size
                        if t == T - 1:
                            del self.audio_memory_bank
                            del self.audio_compression_size
                        elif self.audio_memory_bank.size(1) > self.audio_memory_bank_length:
                            self.audio_memory_bank, self.audio_compression_size = memory_bank_compress(self.audio_memory_bank, self.audio_compression_size)

                else:
                    # logging.info("=====Don't Use Memory Bank Audio=====")
                    audio_query_tokens = self.audio_query_tokens.expand(B, -1, -1)
                    query_atts_audio = torch.ones(audio_query_tokens.size()[:-1], dtype=torch.long).to(audio.device)
                    Qformer_atts_audio = torch.cat([query_atts_audio, text_Qformer_audio["attention_mask"]], dim=1)


                    # 方法一
                    audio_embeds_avg = None
                    audio_atts_avg = None

                    for t in range(T):
                        # Obtain audio embedding for each time step
                        with self.maybe_autocast():
                            audio_embeds = self.ln_audio(self.audio_encoder(audio[:, :, t, :]))  # [B, 256+1, C] (adjust accordingly for audio model)
                        N, C = audio_embeds.shape[-2:]

                        position_ids_audio = torch.tensor([t]).long().to(audio_embeds.device)  # [1]
                        position_ids_audio = position_ids_audio.unsqueeze(0).expand(B, -1)  # [B, 1]
                        audio_embeds = audio_embeds + self.audio_pe(position_ids_audio)  # [B, N, C]

                        audio_atts = torch.ones(audio_embeds.size()[:-1], dtype=torch.long).to(audio.device)  # [B, N]

                        if audio_embeds_avg is None:
                            audio_embeds_avg = audio_embeds.unsqueeze(1)  # [B, 1, N, C]
                            audio_atts_avg = audio_atts.unsqueeze(1)  # [B, 1, N]
                        else:
                            audio_embeds_avg += audio_embeds.unsqueeze(1)  # [B, T, N, C]
                            audio_atts_avg += audio_atts.unsqueeze(1)  # [B, T, N]

                    audio_atts_avg = audio_atts_avg.float()
                    audio_embeds_avg /= T # [B, T, N, C]
                    audio_atts_avg /= T  # [B, T, N]
                    audio_atts_avg = audio_atts_avg.long()


                    # 方法二
                    # audio_embeds_avg = None
                    # audio_atts_avg = None

                    # soft_weights = torch.softmax(self.audio_weights, dim=0).to(audio.device)

                    # for t in range(T):
                    #     # Obtain audio embedding for each time step
                    #     with self.maybe_autocast():
                    #         audio_embeds = self.ln_audio(self.audio_encoder(audio[:, :, t, :]))  # [B, 256+1, C] (adjust accordingly for audio model)

                    #     N, C = audio_embeds.shape[-2:]

                    #     position_ids_audio = torch.tensor([t]).long().to(audio_embeds.device)  # [1]
                    #     position_ids_audio = position_ids_audio.unsqueeze(0).expand(B, -1)  # [B, 1]
                    #     audio_embeds = audio_embeds + self.audio_pe(position_ids_audio)  # [B, N, C]

                    #     audio_atts = torch.ones(audio_embeds.size()[:-1], dtype=torch.long).to(audio.device)  # [B, N]

                    #     if audio_embeds_avg is None:
                    #         audio_embeds_avg = audio_embeds.unsqueeze(1)* soft_weights[t]  # [B, 1, N, C]
                    #         audio_atts_avg = audio_atts.unsqueeze(1)* soft_weights[t]  # [B, 1, N]
                    #     else:
                    #         audio_embeds_avg += audio_embeds.unsqueeze(1)* soft_weights[t]  # [B, T, N, C]
                    #         audio_atts_avg += audio_atts.unsqueeze(1)* soft_weights[t]  # [B, T, N]

                    # audio_atts_avg = audio_atts_avg.float()
                    # audio_embeds_avg /= soft_weights.sum()  # [B, T, N, C]
                    # audio_atts_avg /= soft_weights.sum()  # [B, T, N]
                    # audio_atts_avg = audio_atts_avg.long()

                    # 方法三
                    # audio_embeds_list = []  # 保存每帧的图像特征，后续拼接
                    # audio_atts_list = []

                    # for t in range(T):
                    #     with self.maybe_autocast():
                    #         audio_embeds = self.ln_audio(self.audio_encoder(audio[:, :, t, :]))  # [B, N, C]

                    #     N, C = audio_embeds.shape[-2:]

                    #     # 添加位置编码
                    #     position_ids = torch.tensor([t], device=audio.device).unsqueeze(0).expand(B, -1)
                    #     audio_embeds = audio_embeds + self.audio_pe(position_ids)

                    #     audio_embeds_list.append(audio_embeds)  # [B, N, C]
                    #     audio_atts = torch.ones(audio_embeds.size()[:-1], dtype=torch.long).to(audio.device)  # [B, N]
                    #     audio_atts_list.append(audio_atts)

                    # # 将 T 帧堆叠起来: [B, T, N, C]
                    # audio_embeds_stack = torch.stack(audio_embeds_list, dim=1)  # [B, T, N, C]

                    # audio_embeds_mha = audio_embeds_stack.permute(0, 2, 1, 3).reshape(B * N, T, C)  # [B*N, T, C]

                    # self.temporal_attn_audio.to(audio.device)

                    # # 使用 self-attention 融合时间信息（query=key=value）
                    # attn_out, _ = self.temporal_attn_audio(audio_embeds_mha, audio_embeds_mha, audio_embeds_mha)  # [B*N, T, C]

                    # # 聚合时间维度（平均或取第一个也可以）
                    # attn_out = attn_out.mean(dim=1)  # [B*N, C]

                    # # 恢复到 [B, N, C]
                    # audio_embeds_avg = attn_out.view(B, N, C).unsqueeze(1)

                    # audio_atts_stack = torch.stack(audio_atts_list, dim=1)
                    # audio_atts_avg = audio_atts_stack.float().mean(dim=1, keepdim=True)
                    # audio_atts_avg = (audio_atts_avg > 0.5).long()


                    query_output_audio = self.audio_Qformer.bert(
                        text_Qformer_audio["input_ids"],
                        attention_mask=Qformer_atts_audio,
                        query_embeds=audio_query_tokens,
                        encoder_hidden_states=audio_embeds_avg.squeeze(1),  # Flatten for the batch dimension [B*T, N, C]
                        encoder_attention_mask=audio_atts_avg.squeeze(1),  # Flatten for the batch dimension [B*T, N]
                        return_dict=True,
                    )
            else:
                audio_query_tokens = self.audio_query_tokens.expand(B * T, -1, -1)
                query_output_audio = self.audio_Qformer.bert(
                    query_embeds=audio_query_tokens,
                    encoder_hidden_states=audio_embeds,
                    encoder_attention_mask=audio_atts,
                    return_dict=True,
                )

            inputs_llm_audio = self.llm_proj_audio(query_output_audio.last_hidden_state[:, :audio_query_tokens.size(1), :])
            if self.use_audio:
                inputs_llm_audio = inputs_llm_audio.reshape(B, -1, inputs_llm_audio.shape[-1])
            atts_llm_audio = torch.ones(inputs_llm_audio.size()[:-1], dtype=torch.long).to(image.device)

        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = 'left'
        # print("LLM Input", samples["text_input"])
        text_input_tokens = self.llm_tokenizer(
            samples['text_input'],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(image.device)

        self.llm_tokenizer.truncation_side = 'right'
        # print("eos_token", self.llm_tokenizer.eos_token)
        # print(self.llm_tokenizer.decode(llm_tokens['input_ids'].tolist()[0]))
        self.llm_tokenizer.truncation_side = 'right'
        # prefix = self.llm_tokenizer.eos_token if self.qwen else ""
        # suffix = self.llm_tokenizer.eos_token
        prefix = "<|user|>" if self.qwen else ""
        suffix = "<|user|>"
        text_output_tokens = self.llm_tokenizer(
            [prefix + t + suffix for t in samples['text_output']],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_output_txt_len,
            add_special_tokens=False,
        ).to(image.device)


        llm_tokens, input_part_targets_len = self.concat_text_input_output(
            text_input_tokens.input_ids,
            text_input_tokens.attention_mask,
            text_output_tokens.input_ids,
            text_output_tokens.attention_mask,
        )

        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
        attention_mask = llm_tokens['attention_mask']

        # do not apply loss to the padding
        targets = llm_tokens['input_ids'].masked_fill(
            llm_tokens['input_ids'] == self.llm_tokenizer.pad_token_id, -100
        )

        # do not apply loss to the text input (i.e., instruction)
        for i, l in enumerate(input_part_targets_len):
            targets[i][:l] = -100

        # do not apply loss to the query tokens\
        # change
        # learnable_tokens_embeds = self.learnable_token_embeddings.unsqueeze(0).expand(inputs_embeds.size(0), -1, -1)
        # learnable_tokens_mask = torch.ones(inputs_embeds.size(0), learnable_tokens_embeds.size(1), device=inputs_embeds.device)
        # empty_targets_learnable_tokens = (
        #         torch.ones(learnable_tokens_mask.size(), dtype=torch.long).to(image.device).fill_(-100)
        # )
        # targets = torch.cat([empty_targets_learnable_tokens, targets], dim=1)
        # inputs_embeds = torch.cat([learnable_tokens_embeds, inputs_embeds], dim=1)
        # attention_mask = torch.cat([learnable_tokens_mask, attention_mask], dim=1)

        if self.use_audio:
            empty_targets_audio = (
                torch.ones(atts_llm_audio.size(), dtype=torch.long).to(image.device).fill_(-100)
            )
            # print(f"atts_llm_audio.size: {atts_llm_audio.size()}")
            targets = torch.cat([empty_targets_audio, targets], dim=1)
            inputs_embeds = torch.cat([inputs_llm_audio, inputs_embeds], dim=1)
            attention_mask = torch.cat([atts_llm_audio, attention_mask], dim=1)

        if self.use_video:
            empty_targets = (
                torch.ones(atts_llm.size(), dtype=torch.long).to(image.device).fill_(-100)
            )
            # print(f"atts_llm.size: {atts_llm.size()}")
            targets = torch.cat([empty_targets, targets], dim=1)
            inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
            attention_mask = torch.cat([atts_llm, attention_mask], dim=1)


        with self.maybe_autocast():
            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        # print(outputs.logits)
        loss = outputs.loss

        return {"loss": loss}

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=True, # change
        num_beams=5,
        max_length=128, # 512
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1,
        num_captions=1,
        temperature=1,
    ):

        self.llm_tokenizer.padding_side = "left"

        prompt = samples["prompt"]
        # print("Generate Input", prompt)
        image = samples["image"]
        audio = samples["audio"]
        B = image.size(0)
        # For video data
        is_video = self.use_video
        is_audio = self.use_audio
        if self.use_video and image.dim() == 5:
            B, C, T, H, W = image.shape
        if self.use_audio and audio.dim() == 4:
            B, F, T, C = audio.shape
        if isinstance(prompt, str):
            prompt = [prompt] * B
        else:
            assert len(prompt) == B, "The number of prompts must be equal to the batch size."

        # For TextCaps
        # if "ocr_tokens" in samples.keys() and "{}" in prompt[0]:
        #     prompt = [p.format(', '.join(samples['ocr_tokens'][i][:30])) for i, p in enumerate(prompt)]

        if self.use_audio or self.use_video:
            text_Qformer = self.tokenizer(
                prompt,  # [B]
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            )
            text_Qformer_audio = copy.deepcopy(text_Qformer)
            text_Qformer = {key: value.to(image.device) if isinstance(value, torch.Tensor) else value for key, value in text_Qformer.items()}
            text_Qformer_audio = {key: value.to(audio.device) if isinstance(value, torch.Tensor) else value for key, value in text_Qformer_audio.items()}

        if self.use_video:
            assert self.qformer_text_input == True
            if self.qformer_text_input:
                # remove ocr tokens in q_former (for eval textvqa)
                # qformer_prompt = prompt
                # qformer_prompt = ['Question: ' + qp.split(' Question: ')[1] for qp in qformer_prompt]
                if is_video:
                    text_input = []
                    for text in prompt:
                        text_input.extend([text] * T)
                else:
                    text_input = prompt

                if self.use_memory_bank:
                    query_tokens = self.query_tokens.expand(B, -1, -1)  # [B, 32, C]
                    query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)  # [B, 32]
                    Qformer_atts = torch.cat([query_atts, text_Qformer["attention_mask"]], dim=1)

                    for t in range(T):
                        with self.maybe_autocast():
                            image_embeds = self.ln_vision(self.visual_encoder(image[:, :, t, :, :])).detach()  # [B, 256+1, 1408]
                        N, C = image_embeds.shape[-2:]
                        position_ids = torch.tensor([t]).long().to(image_embeds.device)  # [1]
                        position_ids = position_ids.unsqueeze(0).expand(B, -1)  # [B, 1]
                        image_pe = self.image_pe(position_ids)  # [B, 1, C]
                        image_embeds = image_embeds + image_pe  # [B, N, C]
                        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)  # [B, N]
                        image_embeds = image_embeds.unsqueeze(1)  # [B, 1, N, C]

                        if t == 0:
                            self.visual_memory_bank = image_embeds  # [B, 1, N, C]
                            self.size_constant = torch.ones(B, 1, N).to(image_embeds.device)  # [B, 1, N]
                            self.compression_size = self.size_constant  # [B, 1, N]
                        else:
                            self.visual_memory_bank = torch.cat([self.visual_memory_bank, image_embeds], dim=1)  # [B, t+1, N, C]
                            self.compression_size = torch.cat([self.compression_size, self.size_constant], dim=1)  # [B, t+1, N]

                        query_output = self.Qformer.bert(
                            text_Qformer["input_ids"],
                            attention_mask=Qformer_atts,
                            query_embeds=query_tokens,
                            encoder_hidden_states=self.visual_memory_bank.view(B, -1, C),
                            encoder_attention_mask=image_atts,
                            return_dict=True,
                        )
                        last_hidden_state = query_output.last_hidden_state

                        if t == T - 1:
                            del self.visual_memory_bank
                            del self.compression_size
                        elif self.visual_memory_bank.size(1) > self.memory_bank_length:
                            self.visual_memory_bank, self.compression_size = memory_bank_compress(self.visual_memory_bank,
                                                                                                self.compression_size)
                else:
                    query_tokens = self.query_tokens.expand(B, -1, -1)
                    query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
                    Qformer_atts = torch.cat([query_atts, text_Qformer["attention_mask"]], dim=1)

                    # 方法一
                    image_embeds_avg = None
                    image_atts_avg = None
                    for t in range(T):
                        with self.maybe_autocast():
                            image_embeds = self.ln_vision(self.visual_encoder(image[:, :, t, :, :]))  # [B, 256+1, 1408]
                        N, C = image_embeds.shape[-2:]

                        position_ids = torch.tensor([t]).long().to(image_embeds.device)  # [1]
                        position_ids = position_ids.unsqueeze(0).expand(B, -1)  # [B, 1]
                        image_embeds = image_embeds + self.image_pe(position_ids)  # [B, N, C]

                        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)  # [B, N]

                        if image_embeds_avg is None:
                            image_embeds_avg = image_embeds.unsqueeze(1)  # [B, 1, N, C]
                            image_atts_avg = image_atts.unsqueeze(1)  # [B, 1, N]
                        else:
                            image_embeds_avg += image_embeds.unsqueeze(1)  # [B, T, N, C]
                            image_atts_avg += image_atts.unsqueeze(1)  # [B, T, N]

                    image_atts_avg = image_atts_avg.float()
                    image_embeds_avg /= T  # [B, T, N, C]
                    image_atts_avg /= T  # [B, T, N]
                    image_atts_avg = image_atts_avg.long()

                    # 方法二
                    # image_embeds_avg = None
                    # image_atts_avg = None
                    # soft_weights = torch.softmax(self.weights, dim=0).to(image.device)
                    # for t in range(T):
                    #     with self.maybe_autocast():
                    #         image_embeds = self.ln_vision(self.visual_encoder(image[:, :, t, :, :]))  # [B, 256+1, 1408]
                    #     N, C = image_embeds.shape[-2:]

                    #     position_ids = torch.tensor([t]).long().to(image_embeds.device)  # [1]
                    #     position_ids = position_ids.unsqueeze(0).expand(B, -1)  # [B, 1]
                    #     image_embeds = image_embeds + self.image_pe(position_ids)  # [B, N, C]

                    #     image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)  # [B, N]

                    #     if image_embeds_avg is None:
                    #         image_embeds_avg = image_embeds.unsqueeze(1)* soft_weights[t]  # [B, 1, N, C]
                    #         image_atts_avg = image_atts.unsqueeze(1)* soft_weights[t]  # [B, 1, N]
                    #     else:
                    #         image_embeds_avg += image_embeds.unsqueeze(1)* soft_weights[t]  # [B, T, N, C]
                    #         image_atts_avg += image_atts.unsqueeze(1)* soft_weights[t]  # [B, T, N]

                    # image_atts_avg = image_atts_avg.float()
                    # image_embeds_avg /= soft_weights.sum()  # [B, T, N, C]
                    # image_atts_avg /= soft_weights.sum()  # [B, T, N]
                    # image_atts_avg = image_atts_avg.long()

                    # 方法三
                    # image_embeds_list = []  # 保存每帧的图像特征，后续拼接
                    # image_atts_list = []

                    # for t in range(T):
                    #     with self.maybe_autocast():
                    #         image_embeds = self.ln_vision(self.visual_encoder(image[:, :, t, :, :]))  # [B, N, C]

                    #     N, C = image_embeds.shape[-2:]

                    #     # 添加位置编码
                    #     position_ids = torch.tensor([t], device=image.device).unsqueeze(0).expand(B, -1)
                    #     image_embeds = image_embeds + self.image_pe(position_ids)

                    #     image_embeds_list.append(image_embeds)  # [B, N, C]
                    #     image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)  # [B, N]
                    #     image_atts_list.append(image_atts)

                    # # 将 T 帧堆叠起来: [B, T, N, C]
                    # image_embeds_stack = torch.stack(image_embeds_list, dim=1)  # [B, T, N, C]

                    # # 准备输入 MHA：合并 B 和 N，为了对每个 patch 跨 T 融合
                    # image_embeds_mha = image_embeds_stack.permute(0, 2, 1, 3).reshape(B * N, T, C)  # [B*N, T, C]

                    # self.temporal_attn.to(image.device)

                    # # 使用 self-attention 融合时间信息（query=key=value）
                    # attn_out, _ = self.temporal_attn(image_embeds_mha, image_embeds_mha, image_embeds_mha)  # [B*N, T, C]

                    # # 聚合时间维度（平均或取第一个也可以）
                    # attn_out = attn_out.mean(dim=1)  # [B*N, C]

                    # # 恢复到 [B, N, C]
                    # image_embeds_avg = attn_out.view(B, N, C).unsqueeze(1)

                    # image_atts_stack = torch.stack(image_atts_list, dim=1)
                    # image_atts_avg = image_atts_stack.float().mean(dim=1, keepdim=True)
                    # image_atts_avg = (image_atts_avg > 0.5).long()

                    query_output = self.Qformer.bert(
                        text_Qformer["input_ids"],
                        attention_mask=Qformer_atts,
                        query_embeds=query_tokens,
                        encoder_hidden_states=image_embeds_avg.squeeze(1),  # Flatten for the batch dimension [B*T, N, C]
                        encoder_attention_mask=image_atts_avg.squeeze(1),  # Flatten for the batch dimension [B*T, N]
                        return_dict=True,
                    )
            else:
                query_tokens = self.query_tokens.expand(B * T, -1, -1)
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )

            inputs_llm = self.llm_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
            if is_video:
                inputs_llm = inputs_llm.reshape(B, -1, inputs_llm.shape[-1])
            atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image.device)

        if self.use_audio:
            assert self.qformer_text_input == True
            if self.qformer_text_input:
                if is_audio:
                    text_input = []
                    for text in prompt:
                        text_input.extend([text] * T)
                else:
                    text_input = prompt

                if self.use_memory_bank_audio:
                    audio_query_tokens = self.audio_query_tokens.expand(B, -1, -1)  # [B, 32, C]
                    query_atts_audio = torch.ones(audio_query_tokens.size()[:-1], dtype=torch.long).to(audio.device)  # [B, 32]
                    Qformer_atts_audio = torch.cat([query_atts_audio, text_Qformer_audio["attention_mask"]], dim=1)

                    for t in range(T):
                        with self.maybe_autocast():
                            audio_embeds = self.ln_audio(self.audio_encoder(audio[:, :, t, :])).detach()  # [B, 256+1, 1408]
                        N, C = audio_embeds.shape[-2:]
                        position_ids_audio = torch.tensor([t]).long().to(audio_embeds.device)  # [1]
                        position_ids_audio = position_ids_audio.unsqueeze(0).expand(B, -1)  # [B, 1]
                        audio_pe = self.audio_pe(position_ids_audio)  # [B, 1, C]
                        audio_embeds = audio_embeds + audio_pe  # [B, N, C]
                        audio_atts = torch.ones(audio_embeds.size()[:-1], dtype=torch.long).to(audio.device)  # [B, N]
                        audio_embeds = audio_embeds.unsqueeze(1)  # [B, 1, N, C]

                        if t == 0:
                            self.audio_memory_bank = audio_embeds  # [B, 1, N, C]
                            self.size_constant = torch.ones(B, 1, N).to(audio_embeds.device)  # [B, 1, N]
                            self.compression_size_audio = self.size_constant  # [B, 1, N]
                        else:
                            self.audio_memory_bank = torch.cat([self.audio_memory_bank, audio_embeds], dim=1)  # [B, t+1, N, C]
                            self.compression_size_audio = torch.cat([self.compression_size_audio, self.size_constant], dim=1)  # [B, t+1, N]

                        query_output_audio = self.audio_Qformer.bert(
                            text_Qformer_audio["input_ids"],
                            attention_mask=Qformer_atts_audio,
                            query_embeds=audio_query_tokens,
                            encoder_hidden_states=self.audio_memory_bank.view(B, -1, C),
                            encoder_attention_mask=audio_atts,
                            return_dict=True,
                        )
                        last_hidden_state_audio = query_output_audio.last_hidden_state

                        if t == T - 1:
                            del self.audio_memory_bank
                            del self.compression_size_audio
                        elif self.audio_memory_bank.size(1) > self.audio_memory_bank_length:
                            self.audio_memory_bank, self.compression_size_audio = memory_bank_compress(self.audio_memory_bank,
                                                                                                self.compression_size_audio)

                else:
                    audio_query_tokens = self.audio_query_tokens.expand(B, -1, -1)
                    query_atts_audio = torch.ones(audio_query_tokens.size()[:-1], dtype=torch.long).to(audio.device)
                    Qformer_atts_audio = torch.cat([query_atts_audio, text_Qformer_audio["attention_mask"]], dim=1)

                    # 方法一
                    audio_embeds_avg = None
                    audio_atts_avg = None
                    for t in range(T):
                        with self.maybe_autocast():
                            audio_embeds = self.ln_audio(self.audio_encoder(audio[:, :, t, :]))  # [B, 256+1, C] (adjust accordingly for audio model)

                        N, C = audio_embeds.shape[-2:]

                        position_ids_audio = torch.tensor([t]).long().to(audio_embeds.device)  # [1]
                        position_ids_audio = position_ids_audio.unsqueeze(0).expand(B, -1)  # [B, 1]
                        audio_embeds = audio_embeds + self.audio_pe(position_ids_audio)  # [B, N, C]

                        audio_atts = torch.ones(audio_embeds.size()[:-1], dtype=torch.long).to(audio.device)  # [B, N]

                        if audio_embeds_avg is None:
                            audio_embeds_avg = audio_embeds.unsqueeze(1)  # [B, 1, N, C]
                            audio_atts_avg = audio_atts.unsqueeze(1)  # [B, 1, N]
                        else:
                            audio_embeds_avg += audio_embeds.unsqueeze(1)  # [B, T, N, C]
                            audio_atts_avg += audio_atts.unsqueeze(1)  # [B, T, N]

                    audio_atts_avg = audio_atts_avg.float()
                    audio_embeds_avg /= T  # [B, T, N, C]
                    audio_atts_avg /= T  # [B, T, N]
                    audio_atts_avg = audio_atts_avg.long()

                    # 方法二
                    # audio_embeds_avg = None
                    # audio_atts_avg = None
                    # soft_weights = torch.softmax(self.audio_weights, dim=0).to(audio.device)
                    # for t in range(T):
                    #     with self.maybe_autocast():
                    #         audio_embeds = self.ln_audio(self.audio_encoder(audio[:, :, t, :]))  # [B, 256+1, C] (adjust accordingly for audio model)

                    #     N, C = audio_embeds.shape[-2:]

                    #     position_ids_audio = torch.tensor([t]).long().to(audio_embeds.device)  # [1]
                    #     position_ids_audio = position_ids_audio.unsqueeze(0).expand(B, -1)  # [B, 1]
                    #     audio_embeds = audio_embeds + self.audio_pe(position_ids_audio)  # [B, N, C]

                    #     audio_atts = torch.ones(audio_embeds.size()[:-1], dtype=torch.long).to(audio.device)  # [B, N]

                    #     if audio_embeds_avg is None:
                    #         audio_embeds_avg = audio_embeds.unsqueeze(1)* soft_weights[t]  # [B, 1, N, C]
                    #         audio_atts_avg = audio_atts.unsqueeze(1)* soft_weights[t]  # [B, 1, N]
                    #     else:
                    #         audio_embeds_avg += audio_embeds.unsqueeze(1)* soft_weights[t]  # [B, T, N, C]
                    #         audio_atts_avg += audio_atts.unsqueeze(1)* soft_weights[t]  # [B, T, N]

                    # audio_atts_avg = audio_atts_avg.float()
                    # audio_embeds_avg /= soft_weights.sum()  # [B, T, N, C]
                    # audio_atts_avg /= soft_weights.sum()  # [B, T, N]
                    # audio_atts_avg = audio_atts_avg.long()

                    # 方法三
                    # audio_embeds_list = []  # 保存每帧的图像特征，后续拼接
                    # audio_atts_list = []

                    # for t in range(T):
                    #     with self.maybe_autocast():
                    #         audio_embeds = self.ln_audio(self.audio_encoder(audio[:, :, t, :]))  # [B, N, C]

                    #     N, C = audio_embeds.shape[-2:]

                    #     # 添加位置编码
                    #     position_ids = torch.tensor([t], device=audio.device).unsqueeze(0).expand(B, -1)
                    #     audio_embeds = audio_embeds + self.audio_pe(position_ids)

                    #     audio_embeds_list.append(audio_embeds)  # [B, N, C]
                    #     audio_atts = torch.ones(audio_embeds.size()[:-1], dtype=torch.long).to(audio.device)  # [B, N]
                    #     audio_atts_list.append(audio_atts)

                    # # 将 T 帧堆叠起来: [B, T, N, C]
                    # audio_embeds_stack = torch.stack(audio_embeds_list, dim=1)  # [B, T, N, C]

                    # # 准备输入 MHA：合并 B 和 N，为了对每个 patch 跨 T 融合
                    # audio_embeds_mha = audio_embeds_stack.permute(0, 2, 1, 3).reshape(B * N, T, C)  # [B*N, T, C]

                    # self.temporal_attn_audio.to(audio.device)

                    # # 使用 self-attention 融合时间信息（query=key=value）
                    # attn_out, _ = self.temporal_attn_audio(audio_embeds_mha, audio_embeds_mha, audio_embeds_mha)  # [B*N, T, C]

                    # # 聚合时间维度（平均或取第一个也可以）
                    # attn_out = attn_out.mean(dim=1)  # [B*N, C]

                    # # 恢复到 [B, N, C]
                    # audio_embeds_avg = attn_out.view(B, N, C).unsqueeze(1)

                    # audio_atts_stack = torch.stack(audio_atts_list, dim=1)
                    # audio_atts_avg = audio_atts_stack.float().mean(dim=1, keepdim=True)
                    # audio_atts_avg = (audio_atts_avg > 0.5).long()

                    query_output_audio = self.audio_Qformer.bert(
                        text_Qformer_audio["input_ids"],
                        attention_mask=Qformer_atts_audio,
                        query_embeds=audio_query_tokens,
                        encoder_hidden_states=audio_embeds_avg.squeeze(1),  # Flatten for the batch dimension [B*T, N, C]
                        encoder_attention_mask=audio_atts_avg.squeeze(1),  # Flatten for the batch dimension [B*T, N]
                        return_dict=True,
                    )
            else:
                audio_query_tokens = self.query_tokens.expand(B * T, -1, -1)
                query_output_audio = self.audio_Qformer.bert(
                    query_embeds=audio_query_tokens,
                    encoder_hidden_states=audio_embeds,
                    encoder_attention_mask=audio_atts,
                    return_dict=True,
                )

            inputs_llm_audio = self.llm_proj_audio(query_output_audio.last_hidden_state[:, :audio_query_tokens.size(1), :])
            if is_audio:
                inputs_llm_audio = inputs_llm_audio.reshape(B, -1, inputs_llm_audio.shape[-1])
            atts_llm_audio = torch.ones(inputs_llm_audio.size()[:-1], dtype=torch.long).to(audio.device)

        llm_tokens = self.llm_tokenizer(
            prompt,
            padding="longest",
            return_tensors="pt",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(image.device)

        with self.maybe_autocast():
            inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens.input_ids)
            attention_mask = llm_tokens.attention_mask

            # learnable_tokens_embeds = self.learnable_token_embeddings.unsqueeze(0).expand(inputs_embeds.size(0), -1, -1)
            # learnable_tokens_mask = torch.ones(inputs_embeds.size(0), learnable_tokens_embeds.size(1), device=inputs_embeds.device)

            # inputs_embeds = torch.cat([learnable_tokens_embeds, inputs_embeds], dim=1)
            # attention_mask = torch.cat([learnable_tokens_mask, attention_mask], dim=1)

            if self.use_audio:
                inputs_embeds = torch.cat([inputs_llm_audio, inputs_embeds], dim=1)
                attention_mask = torch.cat([atts_llm_audio, attention_mask], dim=1)
            if self.use_video:
                inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
                attention_mask = torch.cat([atts_llm, attention_mask], dim=1)

            # outputs = self.llm_model.generate(
            #     inputs_embeds=inputs_embeds,
            #     attention_mask=attention_mask,
            #     do_sample=use_nucleus_sampling,
            #     top_p=0.95, # top_p,
            #     temperature=0.7, # temprature
            #     num_beams=num_beams,
            #     # max_length=max_length,
            #     # min_length=min_length,
            #     # eos_token_id=self.eos_token_id,
            #     repetition_penalty=1.8, # repetition_penalty,
            #     length_penalty=0.5, # length_penalty,
            #     num_return_sequences=num_captions,
            #     max_new_tokens=125 # 16, 32
            # )
            # change
            outputs = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
                num_return_sequences=num_captions,
                max_new_tokens=max_length,
            )

        # outputs[outputs < 2] = 2 # convert output id -1/0/1 to 2 (eos_token_id)
        output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]

        return output_text

    def predict_answers(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=10,
        min_len=1,
        num_ans_candidates=128, # 128 512
        answer_list=None,
        prompt="",
        length_penalty=0,
        **kwargs
    ):
        if isinstance(samples["text_input"], str):
            samples["text_input"] = [samples["text_input"]]

        if prompt:
            if prompt.count("{}") == 2:
                if 'ocr_tokens' in samples:
                    text_input = [
                        prompt.format(', '.join(samples['ocr_tokens'][i][:30]), samples["text_input"][i])
                    for i in range(len(samples["text_input"]))]
                elif 'choices' in samples:
                    text_input = []
                    for i in range(len(samples["text_input"])):
                        this_choices = [f"({string.ascii_lowercase[j]}) {ch}" for j, ch in enumerate(samples["choices"][i])]
                        this_choices = " ".join(this_choices)
                        text_input.append(prompt.format(samples["text_input"][i], this_choices))
            else:
                text_input = [prompt.format(question) for question in samples["text_input"]]
        else:
            text_input = samples["text_input"]

        samples["prompt"] = text_input

        output_text = self.generate(
            samples,
            num_beams=num_beams,
            max_length=max_len,
            min_length=min_len,
            length_penalty=length_penalty,
            num_captions=num_beams,
        )

        if "apply_lemmatizer" in samples.keys() and samples["apply_lemmatizer"]:
            output_text = self._lemmatize(output_text)

        # print("output_text:", output_text)

        return output_text

    def predict_class(
        self,
        samples,
        candidates,
        n_segments=1,
    ):
        self.llm_tokenizer.padding_side = "left"

        # If candidates is a list of lists, each sample has its candidates, then we need to iterate one by one
        if type(candidates[0]) == list:
            results = []

            for i in range(samples["image"].size(0)):
                this_sample = {
                    "image": samples["image"][i].unsqueeze(0),
                    "prompt": samples["prompt"],
                }

                if "text_input" in samples.keys():
                    this_sample["text_input"] = [samples["text_input"][i]]

                if 'context' in samples.keys():
                    this_sample['context'] = [samples["context"][i]]

                if 'history' in samples.keys():
                    this_sample['history'] = [samples["history"][i]]

                if 'caption' in samples.keys():
                    this_sample['caption'] = [samples["caption"][i]]

                this_result = self._predict_class(this_sample, candidates[i], n_segments)
                results.append(this_result)

            try:
                results = torch.cat(results, dim=0)
            except:
                results = [res.tolist()[0] for res in results]

            return results

        return self._predict_class(samples, candidates, n_segments)

    def _predict_class(
        self,
        samples,
        candidates,
        n_segments=1,
    ):
        image = samples["image"]
        prompt = samples["prompt"]

        bs = image.size(0)

        if isinstance(prompt, str):
            prompt = [prompt] * bs
        else:
            assert len(prompt) == bs, "The number of prompts must be equal to the batch size."

        if "text_input" in samples.keys():
            if type(samples["text_input"][0]) == list:
                prompt = [prompt[i].format(*samples["text_input"][i]) for i in range(len(prompt))]
            else:
                prompt = [prompt[i].format(samples["text_input"][i]) for i in range(len(prompt))]

        # scienceqa
        if 'context' in samples.keys() and samples['context'] != '':
            prompt = [f'context: {samples["context"][i]}. {prompt[i]}' for i in range(len(prompt))]

        # visual dialog
        if 'history' in samples.keys() and samples['history'][0] != '':
            prompt = [f'dialog history: {samples["history"][i]}\n{prompt[i]}' for i in range(len(prompt))]

        # change
        if 'caption' in samples.keys() and samples['caption'][0] != '':
            prompt = [f'This image has the caption "{samples["caption"][i]}". {prompt[i]}' for i in range(len(prompt))]

        query_tokens = self.query_tokens.expand(bs, -1, -1)
        if self.qformer_text_input:
            text_Qformer = self.tokenizer(
                prompt,
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt"
            ).to(image.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

        if image.dim() == 5:
            inputs_llm, atts_llm = [], []
            for j in range(image.size(2)):
                this_frame = image[:,:,j,:,:]
                with self.maybe_autocast():
                    frame_embeds = self.ln_vision(self.visual_encoder(this_frame))
                    frame_atts = torch.ones(frame_embeds.size()[:-1], dtype=torch.long).to(image.device)

                if self.qformer_text_input:
                    frame_query_output = self.Qformer.bert(
                        text_Qformer.input_ids,
                        attention_mask=Qformer_atts,
                        query_embeds=query_tokens,
                        encoder_hidden_states=frame_embeds,
                        encoder_attention_mask=frame_atts,
                        return_dict=True,
                    )
                else:
                    frame_query_output = self.Qformer.bert(
                        query_embeds=query_tokens,
                        encoder_hidden_states=frame_embeds,
                        encoder_attention_mask=frame_atts,
                        return_dict=True,
                    )

                frame_inputs_llm = self.llm_proj(frame_query_output.last_hidden_state[:,:query_tokens.size(1),:])
                frame_atts_llm = torch.ones(frame_inputs_llm.size()[:-1], dtype=torch.long).to(image.device)
                inputs_llm.append(frame_inputs_llm)
                atts_llm.append(frame_atts_llm)
            inputs_llm = torch.cat(inputs_llm, dim=1)
            atts_llm = torch.cat(atts_llm, dim=1)
        else:
            with self.maybe_autocast():
                image_embeds = self.ln_vision(self.visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

            if self.qformer_text_input:
                query_output = self.Qformer.bert(
                    text_Qformer.input_ids,
                    attention_mask=Qformer_atts,
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
            else:
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )

            inputs_llm = self.llm_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
            atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image.device)

        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = 'left'
        text_input_tokens = self.llm_tokenizer(
            prompt,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,  # change
        ).to(image.device)

        empty_targets = torch.ones(atts_llm.size(), dtype=torch.long).to(image.device).fill_(-100)

        # self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = 'right'
        n_cands = len(candidates)
        with self.maybe_autocast(dtype=torch.bfloat16):
            all_losses = []
            for n in range(n_segments):
                seg_len = n_cands // n_segments
                if n == (n_segments - 1):
                    seg_len = n_cands - seg_len * (n_segments - 1)

                start_i = n * (n_cands // n_segments)
                end_i = start_i + seg_len

                this_output_tokens = self.llm_tokenizer(
                    candidates[start_i:end_i],
                    return_tensors="pt",
                    padding="longest",
                    truncation=True,
                    max_length=self.max_output_txt_len,  # change
                ).to(image.device)

                this_input_tokens_ids = text_input_tokens.input_ids.repeat_interleave(seg_len, dim=0)
                this_input_tokens_atts = text_input_tokens.attention_mask.repeat_interleave(seg_len, dim=0)

                this_output_tokens_ids = this_output_tokens.input_ids.repeat(bs, 1)
                this_output_tokens_atts = this_output_tokens.attention_mask.repeat(bs, 1)

                this_llm_tokens, this_input_targets_len = self.concat_text_input_output(
                    this_input_tokens_ids,
                    this_input_tokens_atts,
                    this_output_tokens_ids,
                    this_output_tokens_atts
                )

                this_llm_input_ids = this_llm_tokens['input_ids']
                this_llm_atts = this_llm_tokens['attention_mask']
                # this_llm_input_ids = torch.cat([this_input_tokens_ids, this_output_tokens_ids], dim=1)
                # this_llm_atts = torch.cat([this_input_tokens_atts, this_output_tokens_atts], dim=1)

                inputs_embeds = self.llm_model.get_input_embeddings()(this_llm_input_ids)
                inputs_embeds = torch.cat([inputs_llm.repeat_interleave(seg_len, dim=0), inputs_embeds], dim=1)
                attention_mask = torch.cat([atts_llm.repeat_interleave(seg_len, dim=0), this_llm_atts], dim=1)

                this_targets = this_llm_input_ids.masked_fill(this_llm_input_ids == self.llm_tokenizer.pad_token_id, -100)
                # this_targets[:, :this_input_tokens_ids.size(1)] = -100
                for i, l in enumerate(this_input_targets_len):
                    this_targets[i][:l] = -100

                this_targets = torch.cat([empty_targets.repeat_interleave(seg_len, dim=0), this_targets], dim=1)

                outputs = self.llm_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=this_targets,
                    reduction="none",
                )

                loss = outputs.loss

                loss = loss.reshape(bs, seg_len)
                # output_class_ranks = torch.argsort(loss, dim=-1)
                all_losses.append(loss)

            all_losses = torch.cat(all_losses, dim=-1)
            output_class_ranks = torch.argsort(all_losses, dim=-1)

        return output_class_ranks

    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llm_model = cfg.get("llm_model")
        bert_path = cfg.get("bert_path")
        memory_bank_length = cfg.get("memory_bank_length", 0)
        audio_memory_bank_length = cfg.get("audio_memory_bank_length", 0)
        num_frames = cfg.get("num_frames", 0)

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        # max_txt_len = cfg.get("max_txt_len", 128)
        max_txt_len = cfg.get("max_txt_len", 128) # 512
        # max_output_txt_len = cfg.get("max_output_txt_len", 256)
        max_output_txt_len = cfg.get("max_output_txt_len", 32) # 32
        use_audio=cfg.get("use_audio", True)
        use_video=cfg.get("use_video", True)
        use_lora = cfg.get("use_lora", True)
        use_qlora = cfg.get("use_qlora", True)
        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        qformer_text_input = cfg.get("qformer_text_input", True)
        qwen = cfg.get("qwen", True)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            llm_model=llm_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            max_output_txt_len=max_output_txt_len,
            apply_lemmatizer=apply_lemmatizer,
            qformer_text_input=qformer_text_input,
            memory_bank_length=memory_bank_length,
            audio_memory_bank_length=audio_memory_bank_length,
            num_frames=num_frames,
            use_audio=use_audio,
            use_video=use_video,
            use_LoRA=use_lora,
            use_QLoRA=use_qlora,
            bert_path=bert_path,
            qwen=qwen,
        )

        # if qformer_text_input:
        #     # Hard-coded to load from BLIP-2 stage-1 pre-trained model (not ideal)
        #     model.load_from_pretrained(
        #         url_or_filename="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained.pth"
        #     )

        model.load_checkpoint_from_config(cfg)

        return model

import math
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

from vispeak.constants import AUDIO_TOKEN_INDEX, IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_SEG_TOKEN, VIDEO_AUDIO_TOKEN_INDEX, DEFAULT_BOS_TOKEN, DEFAULT_IMAGE_TOKEN_NUMBER, DEFAULT_AUDIO_TOKEN_NUMBER

from .multimodal_encoder.builder import build_audio_encoder, build_vision_tower
from .multimodal_projector.builder import build_vision_projector
import numpy as np

class ViSpeakMetaModel:
    def __init__(self, config):
        super(ViSpeakMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(
                config, delay_load=False#not getattr(config, "continuous_training", False)
            )
            if getattr(config, "continuous_training", False):
                config.continuous_training = False
            self.mm_projector = build_vision_projector(config)

        if hasattr(config, "mm_audio_encoder"):
            self.audio_encoder = build_audio_encoder(config)
        
        
        if hasattr(config, "combination"):
            self.combination_type = config.combination
            if getattr(config, "combination") == 'linear_combine':
                self.combination = torch.nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
            elif getattr(config, "combination") == 'adaptive_combine':
                self.combination = torch.nn.Sequential(
                    torch.nn.Linear(config.hidden_size * 2, config.hidden_size),
                    torch.nn.ReLU(),
                    torch.nn.Linear(config.hidden_size, 1,),
                )

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def get_audio_encoder(self):
        audio_encoder = getattr(self, "audio_encoder", None)
        return audio_encoder
    
    def get_combination(self):
        combination = getattr(self, "combination", None)
        return combination

    def initialize_vision_modules(self, model_args):
        vision_tower = model_args.vision_tower

        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)
            self.vision_tower = vision_tower
        else:
            vision_tower = self.vision_tower
            #vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, "mm_projector_type")
        self.config.mm_hidden_size = vision_tower.hidden_size

        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, "mm_projector"))

    def initialize_audio_modules(self, model_args):
        audio_encoder = model_args.audio_encoder

        pretrain_audio_mlp_adapter = model_args.pretrain_audio_mlp_adapter

        setattr(self.config, "mm_audio_encoder", audio_encoder)

        audio_encoder = build_audio_encoder(self.config)
        self.audio_encoder = audio_encoder

        load_audio_ckpt_from_mllm = True
        if load_audio_ckpt_from_mllm:
            from safetensors.torch import load_file
            import os
            audio_weights = {}
            for file_name in os.listdir(model_args.model_name_or_path):
                if file_name.endswith('safetensors'):
                    audio_weights.update(
                        {k[20:]: v for k, v in load_file(os.path.join(model_args.model_name_or_path, file_name)).items() if
                            k.startswith('model.audio_encoder.')})
            self.audio_encoder.load_state_dict(audio_weights, strict=True) 
            print("load audio encoder from mllm")
            print("load audio encoder Done!")

        #load_audio_ckpt = True
        #if self.get_audio_encoder() is None or load_audio_ckpt or model_args.audio_prompt_finetune:
        #    audio_encoder = build_audio_encoder(self.config)
        #    self.audio_encoder = audio_encoder

        #load_audio_prompt_weight = False #True
        #if load_audio_prompt_weight:
        #    from safetensors.torch import load_file
        #    import os
        #    audio_weights = {}
        #    for file_name in os.listdir(model_args.model_name_or_path):
        #        if file_name.endswith('safetensors'):
        #            audio_weights.update(
        #                {k[38:]: v for k, v in load_file(os.path.join(model_args.model_name_or_path, file_name)).items() if
        #                    k.startswith('model.audio_encoder.prompt_embeddings')})
        #    self.audio_encoder.prompt_embeddings.load_state_dict(audio_weights, strict=True)

        #checkpoint = torch.load(model_args.audio_encoder + "/final.pt", map_location="cpu")
        #model_dict = self.audio_encoder.state_dict()
        #for key in model_dict.keys():
        #    if key in checkpoint.keys():
        #        if model_dict[key].shape == checkpoint[key].shape:
        #            model_dict[key] = checkpoint[key]
        #        else:
        #            print(
        #                "Key {} has different shape, {} VS {}".format(
        #                    key, model_dict[key].shape, checkpoint[key].shape
        #                )
        #            )
        #    else:
        #        print("Key {} has not in resume model".format(key))
        #self.audio_encoder.load_state_dict(model_dict)

        if pretrain_audio_mlp_adapter is not None:
            audio_projector_weights = torch.load(pretrain_audio_mlp_adapter, map_location="cpu")

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            self.audio_encoder.adpter.load_state_dict(get_w(audio_projector_weights, "audio_encoder.adpter"))
    
    def initialize_combination_modules(self, model_args):
        pretrain_combination_modules = model_args.pretrain_combination_modules
        self.config.combination = model_args.combination

        if getattr(self, "combination", None) is None:
            if self.config.combination == 'linear_combine':
                self.combination = torch.nn.Linear(self.config.hidden_size * 2, self.config.hidden_size, bias=False)
                # 重置权重矩阵，使前半部分为单位矩阵，后半部分为零
                self.combination.weight.data[:, :self.config.hidden_size] = torch.eye(self.config.hidden_size)
                self.combination.weight.data[:, self.config.hidden_size:] = 0
            elif self.config.combination == 'adaptive_combine':
                self.combination = torch.nn.Sequential(
                    torch.nn.Linear(self.config.hidden_size * 2, self.config.hidden_size),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.config.hidden_size, 1,),
                )
            
        else:
            # In case it is frozen by LoRA
            for p in self.combination.parameters():
                p.requires_grad = True

        if pretrain_combination_modules is not None:
            mm_projector_weights = torch.load(pretrain_combination_modules, map_location="cpu")

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            self.combination.load_state_dict(get_w(mm_projector_weights, "combination"))


class ViSpeakMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_audio_encoder(self):
        return self.get_model().get_audio_encoder()

    def pool_feats(self, x, out_size):
        ndim = x.ndim
        if ndim == 2:
            x = x.unsqueeze(0)
        b, num_tokens, c = x.shape
        h = int(math.sqrt(num_tokens))
        x = x.permute(0, 2, 1).reshape(b, -1, h, h)
        x = F.interpolate(x, size=out_size, mode='bilinear', align_corners=False)
        num_tokens = x.shape[2] * x.shape[3]  # Recalculate the number of tokens after pooling
        x = x.reshape(b, c, num_tokens).permute(0, 2, 1)
        if ndim == 2:
            x = x.squeeze(0)
        return x

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        pooling_size = getattr(self.config, "pooling_size", 1)
        if pooling_size > 1:
            h = int(math.sqrt(image_features.shape[1]))
            image_features = self.pool_feats(image_features, (h // pooling_size, h // pooling_size))
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def encode_images_frameCat(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        assert len(image_features) % 5 == 0

        concatenated_features = []
        for i in range(0, len(image_features), 5):
            tensors_to_concat = [image_features[j] for j in range(i, i + 5)]
            concatenated_tensor = torch.cat(tensors_to_concat, dim=-1)
            concatenated_features.append(concatenated_tensor)
        concatenated_features = torch.stack(concatenated_features)
        image_features = concatenated_features

        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def slow_fast_pooling0(self, temp_img_feats):
        num_frame = len(temp_img_feats)
        if num_frame <= 30:
            slow_token_num = max([e for e in [256, 225, 196, 169] if e <= 5200/num_frame]) 
            fast_token_num = slow_token_num
        elif num_frame <= 45:
            slow_token_num = 169
            fast_token_num = 81
        elif num_frame <= 64:
            slow_token_num = 169
            fast_token_num = 49
        else:
            raise ValueError("The number of frames is too large!")
        
        if num_frame <= 30:
            num_slow = num_frame
        else:
            num_slow = int((5200 - fast_token_num * num_frame) / (slow_token_num - fast_token_num))
        num_fast = num_frame - num_slow
        slow_index = list(np.linspace(0, num_frame, num=num_slow, dtype=int))

        new_img_feats = []
        for i, feat in enumerate(temp_img_feats):
            if i in slow_index:
                sqrt_len = int(math.sqrt(slow_token_num))
            else:
                sqrt_len = int(math.sqrt(fast_token_num))
            if sqrt_len != 16:
                feat = self.pool_feats(feat, out_size=(sqrt_len, sqrt_len))
            new_img_feats.append(feat)

        return new_img_feats

    def slow_fast_pooling1(self, temp_img_feats):
        num_frame = len(temp_img_feats)
        if num_frame <= 28:
            slow_token_num = max([e for e in [256, 225, 196, 169, 144] if e <= 4096/num_frame]) 
            fast_token_num = slow_token_num
        elif num_frame <= 40:
            slow_token_num = 144
            fast_token_num = 81
        elif num_frame <= 64:
            slow_token_num = 144
            fast_token_num = 49
        else:
            raise ValueError("The number of frames is too large!")
        
        if num_frame <= 28:
            num_slow = num_frame
        else:
            num_slow = int((4096 - fast_token_num * num_frame) / (slow_token_num - fast_token_num))
        num_fast = num_frame - num_slow
        slow_index = list(np.linspace(0, num_frame, num=num_slow, dtype=int))

        new_img_feats = []
        for i, feat in enumerate(temp_img_feats):
            if i in slow_index:
                sqrt_len = int(math.sqrt(slow_token_num))
            else:
                sqrt_len = int(math.sqrt(fast_token_num))
            if sqrt_len != 16:
                feat = self.pool_feats(feat, out_size=(sqrt_len, sqrt_len))
            new_img_feats.append(feat)

        return new_img_feats

    def slow_fast_pooling(self, temp_img_feats):
        num_frame = len(temp_img_feats)
        slow_token_num = 144
        fast_token_num = 49
        
        slow_index = list(range(0, num_frame, 4))

        new_img_feats = []
        for i, feat in enumerate(temp_img_feats):
            if i in slow_index:
                sqrt_len = int(math.sqrt(slow_token_num))
            else:
                sqrt_len = int(math.sqrt(fast_token_num))
            if sqrt_len != 16:
                feat = self.pool_feats(feat, out_size=(sqrt_len, sqrt_len))
            new_img_feats.append(feat)

        return new_img_feats

    def slow_fast_pooling3(self, temp_img_feats):
        num_frame = len(temp_img_feats)
        slow_token_num = 144
        fast_token_num = 36
        
        slow_index = list(range(0, num_frame, 16))

        new_img_feats = []
        for i, feat in enumerate(temp_img_feats):
            if i in slow_index:
                sqrt_len = int(math.sqrt(slow_token_num))
            else:
                sqrt_len = int(math.sqrt(fast_token_num))
            if sqrt_len != 16:
                feat = self.pool_feats(feat, out_size=(sqrt_len, sqrt_len))
            new_img_feats.append(feat)

        return new_img_feats

    def slow_fast(self, image_features, sf_masks):
        new_image_features = []
        temp_img_feats = []  # 初始化 temp_img_feats 在循环外
        for i, img_feat in enumerate(image_features):
            if i == 0 or sf_masks[i] != sf_masks[i-1]:
                if temp_img_feats:  # 如果 temp_img_feats 不为空，则添加到 new_image_features
                    if sf_masks[i-1] > 0:
                        temp_img_feats = self.slow_fast_pooling(temp_img_feats)
                    new_image_features.append(temp_img_feats)
                temp_img_feats = [img_feat]  # 重新初始化 temp_img_feats
            else:
                temp_img_feats.append(img_feat)
        if temp_img_feats:  # 处理最后一个子列表
            if sf_masks[-1] > 0:
                temp_img_feats = self.slow_fast_pooling(temp_img_feats)
            new_image_features.append(temp_img_feats)
        
        output_features = []
        for e in new_image_features:
            output_features += e

        return output_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, user_input_ids, position_ids, attention_mask, past_key_values, labels, images, audios, video_audios, pad_token_id, sf_masks, shared_v_pid_stride=None, informative_labels=None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if (
                past_key_values is not None
                and vision_tower is not None
                and images is not None
                and input_ids.shape[1] == 1
            ):
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat(
                    (
                        attention_mask,
                        torch.ones(
                            (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        ),
                    ),
                    dim=1,
                )
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None

        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1).to(self.device) for x in image_features]
        else:
            image_features = self.encode_images(images).to(self.device)

        image_features = [e for e in image_features]
        if sf_masks is not None:
            assert len(image_features) == len(sf_masks)
            image_features = self.slow_fast(image_features, sf_masks) 

        audio_encoder = self.get_audio_encoder()
        if audios is not None:
            audio_features = audio_encoder(audios["audios"], audios["lengths"])
            state_labels = audios.get("state_labels", None)
            lengths_for_llm = audios["lengths_for_llm"]
            if state_labels is not None:
                assert len(audio_features["inputs_embeds"]) == len(state_labels) == len(lengths_for_llm)
        else:
            audio_features, state_labels, lengths_for_llm = None, None, None        

        if video_audios is not None:
            video_audio_features = audio_encoder(video_audios["audios"], video_audios["lengths"])
            video_audio_state_labels = video_audios.get("state_labels", None)
            video_audio_lengths_for_llm = video_audios["lengths_for_llm"]
            if video_audio_state_labels is not None:
                assert len(video_audio_features["inputs_embeds"]) == len(video_audio_state_labels) == len(video_audio_lengths_for_llm)
        else:
            video_audio_features, video_audio_state_labels, video_audio_lengths_for_llm = None, None, None  

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        _informative_labels = informative_labels
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            )
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)
        if informative_labels is None:
            informative_labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- TODO: double check
        input_ids = [
            cur_input_ids[cur_attention_mask]
            for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
        ]
        user_input_ids = [
            cur_input_ids[cur_attention_mask]
            for cur_input_ids, cur_attention_mask in zip(user_input_ids, attention_mask)
        ]
        labels = [
            cur_labels[cur_attention_mask]
            for cur_labels, cur_attention_mask in zip(labels, attention_mask)
        ]
        informative_labels = [
            cur_labels[cur_attention_mask]
            for cur_labels, cur_attention_mask in zip(informative_labels, attention_mask)
        ]

        new_input_embeds = []
        new_user_input_embeds = []
        new_labels = []
        new_informative_labels = []
        v_start_end = []
        cur_image_idx = 0
        cur_audio_idx = 0
        cur_video_audio_idx = 0
        pooling_size = getattr(self.config, "pooling_size", 1)
        single_image_tokens = DEFAULT_IMAGE_TOKEN_NUMBER // pooling_size // pooling_size
        # if not (sum([(cur == IMAGE_TOKEN_INDEX).sum() // single_image_tokens for cur in user_input_ids]) + sum([(IMAGE_TOKEN_INDEX not in cur) for cur in user_input_ids]) == len(image_features)):
        #     print("#################################################")
        #     num_need = sum([(cur == IMAGE_TOKEN_INDEX).sum() // single_image_tokens for cur in user_input_ids]) + sum([(IMAGE_TOKEN_INDEX not in cur) for cur in user_input_ids])
        #     print(num_need, len(image_features))
        #     torch.save(user_input_ids, 'invalid_user_input_ids.pth')
        #     print("#################################################")
        #     new_labels = []
        #     for lab in labels:
        #         lab = -100
        #         new_labels.append(lab)
        #     labels = new_labels
        #     if num_need < len(image_features):
        #         image_features = image_features[:num_need]
        #     else:
        #         image_features = image_features + image_features[:(num_need - len(image_features))]
        
        # if not (sum([(cur == AUDIO_TOKEN_INDEX).sum() // DEFAULT_AUDIO_TOKEN_NUMBER for cur in user_input_ids]) + sum([(AUDIO_TOKEN_INDEX not in cur) for cur in user_input_ids]) == len(audio_features["inputs_embeds"])):
        #     print("#################################################")
        #     num_need = sum([(cur == AUDIO_TOKEN_INDEX).sum() // DEFAULT_AUDIO_TOKEN_NUMBER for cur in user_input_ids]) + sum([(AUDIO_TOKEN_INDEX not in cur) for cur in user_input_ids])
        #     print(num_need, len(audio_features["inputs_embeds"]))
        #     torch.save(user_input_ids, 'invalid_user_input_ids.pth')
        #     print("#################################################")
        #     new_labels = []
        #     for lab in labels:
        #         lab = -100
        #         new_labels.append(lab)
        #     labels = new_labels
        #     if num_need < len(audio_features["inputs_embeds"]):
        #         audio_features["inputs_embeds"] = audio_features["inputs_embeds"][:num_need]
        #     else:
        #         audio_features["inputs_embeds"] = torch.cat([audio_features["inputs_embeds"], audio_features["inputs_embeds"][:(num_need - len(audio_features["inputs_embeds"]))]])
        
        # if not (sum([(cur == VIDEO_AUDIO_TOKEN_INDEX).sum() // DEFAULT_AUDIO_TOKEN_NUMBER for cur in user_input_ids]) + sum([(VIDEO_AUDIO_TOKEN_INDEX not in cur) for cur in user_input_ids]) == len(video_audio_features["inputs_embeds"])):
        #     print("#################################################")
        #     num_need = sum([(cur == VIDEO_AUDIO_TOKEN_INDEX).sum() // DEFAULT_AUDIO_TOKEN_NUMBER for cur in user_input_ids]) + sum([(VIDEO_AUDIO_TOKEN_INDEX not in cur) for cur in user_input_ids])
        #     print(num_need, len(video_audio_features["inputs_embeds"]))
        #     torch.save(user_input_ids, 'invalid_user_input_ids.pth')
        #     print("#################################################")
        #     new_labels = []
        #     for lab in labels:
        #         lab = -100
        #         new_labels.append(lab)
        #     labels = new_labels
        #     if num_need < len(video_audio_features["inputs_embeds"]):
        #         video_audio_features["inputs_embeds"] = video_audio_features["inputs_embeds"][:num_need]
        #     else:
        #         video_audio_features["inputs_embeds"] = torch.cat([video_audio_features["inputs_embeds"], video_audio_features["inputs_embeds"][:(num_need - len(video_audio_features["inputs_embeds"]))]])

        assert (
            sum([(cur == IMAGE_TOKEN_INDEX).sum() // single_image_tokens for cur in user_input_ids])
            + sum([(IMAGE_TOKEN_INDEX not in cur) for cur in user_input_ids])
            == len(image_features)
        ), user_input_ids
        assert (
            sum([(cur == AUDIO_TOKEN_INDEX).sum() // DEFAULT_AUDIO_TOKEN_NUMBER for cur in user_input_ids])
            + sum([(AUDIO_TOKEN_INDEX not in cur) for cur in user_input_ids])
            == len(audio_features["inputs_embeds"])
        ), user_input_ids
        assert (
            sum([(cur == VIDEO_AUDIO_TOKEN_INDEX).sum() // DEFAULT_AUDIO_TOKEN_NUMBER for cur in user_input_ids])
            + sum([(VIDEO_AUDIO_TOKEN_INDEX not in cur) for cur in user_input_ids])
            == len(video_audio_features["inputs_embeds"])
        ), user_input_ids

        for batch_idx, (cur_input_ids, cur_user_input_ids) in enumerate(zip(input_ids, user_input_ids)):

            num_images = (cur_user_input_ids == IMAGE_TOKEN_INDEX).sum() // single_image_tokens
            num_audio_frames = (cur_user_input_ids == AUDIO_TOKEN_INDEX).sum() // DEFAULT_AUDIO_TOKEN_NUMBER
            num_video_audio_frames = (cur_user_input_ids == VIDEO_AUDIO_TOKEN_INDEX).sum() // DEFAULT_AUDIO_TOKEN_NUMBER
            if num_images == 0 and num_audio_frames == 0 and num_video_audio_frames == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_audio_features = audio_features["inputs_embeds"][cur_audio_idx]
                cur_video_audio_features = video_audio_features["inputs_embeds"][cur_video_audio_idx]
                cur_user_input_embeds_1 = self.get_model().embed_tokens(cur_user_input_ids)
                cur_user_input_embeds = torch.cat(
                    [cur_user_input_embeds_1, cur_image_features[0:0], cur_audio_features[0:0], cur_video_audio_features[0:0]], dim=0
                )
                new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                new_user_input_embeds.append(cur_user_input_embeds)
                new_labels.append(labels[batch_idx])
                new_informative_labels.append(informative_labels[batch_idx])
                cur_image_idx += 1
                cur_audio_idx += 1
                cur_video_audio_idx += 1
                continue

            cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
            cur_labels = labels[batch_idx]
            cur_informative_labels = informative_labels[batch_idx]

            multimodal_embedding_mask = cur_user_input_ids < 0
            multimodal_ids = cur_user_input_ids[multimodal_embedding_mask]
            cur_user_input_ids[multimodal_embedding_mask] = 0
            cur_user_input_embeds = self.get_model().embed_tokens(cur_user_input_ids)
            multimodal_embeddings = []
            point = 0
            while point < len(multimodal_ids):
                if multimodal_ids[point] == IMAGE_TOKEN_INDEX:
                    point += single_image_tokens
                    multimodal_embeddings.append(image_features[cur_image_idx])
                    cur_image_idx += 1
                elif multimodal_ids[point] == AUDIO_TOKEN_INDEX:
                    point += DEFAULT_AUDIO_TOKEN_NUMBER
                    multimodal_embeddings.append(audio_features["inputs_embeds"][cur_audio_idx])
                    cur_audio_idx += 1
                elif multimodal_ids[point] == VIDEO_AUDIO_TOKEN_INDEX:
                    point += DEFAULT_AUDIO_TOKEN_NUMBER
                    multimodal_embeddings.append(video_audio_features["inputs_embeds"][cur_video_audio_idx])
                    cur_video_audio_idx += 1
                else:
                    assert False, "Invalid multimodal token!"
            multimodal_embeddings = torch.cat(multimodal_embeddings)
            new_embeds = cur_user_input_embeds.clone()  # avoid in-place operator
            new_embeds[multimodal_embedding_mask] = multimodal_embeddings
            cur_user_new_input_embeds = [new_embeds]


            if num_audio_frames == 0:
                cur_audio_features = audio_features["inputs_embeds"][cur_audio_idx]
                cur_audio_idx += 1
                cur_user_new_input_embeds.append(cur_audio_features[0:0])
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_image_idx += 1
                cur_user_new_input_embeds.append(cur_image_features[0:0])
            if num_video_audio_frames == 0:
                cur_video_audio_features = video_audio_features["inputs_embeds"][cur_video_audio_idx]
                cur_video_audio_idx += 1
                cur_user_new_input_embeds.append(cur_video_audio_features[0:0])
            cur_user_new_input_embeds = torch.cat(cur_user_new_input_embeds)

            assert len(cur_input_embeds) == len(cur_user_new_input_embeds)
            new_input_embeds.append(cur_input_embeds)
            new_user_input_embeds.append(cur_user_new_input_embeds)
            new_labels.append(cur_labels)
            new_informative_labels.append(cur_informative_labels)


        assert cur_image_idx == len(image_features)
        assert cur_audio_idx == audio_features["inputs_embeds"].shape[0]
        assert cur_video_audio_idx == video_audio_features["inputs_embeds"].shape[0]

        # if state_labels is not None:
        #     assert cur_audio_idx == len(state_labels)
        # if state_labels is not None:
        #     assert (
        #         sum([(cur == AUDIO_TOKEN_INDEX).sum() for cur in input_ids])
        #         == sum([(cur == -101).sum() for cur in new_labels]) + sum([(cur == -102).sum() for cur in new_labels])
        #     ), (input_ids, sum([(cur == AUDIO_TOKEN_INDEX).sum() for cur in input_ids]),  sum([(cur == -101).sum() for cur in new_labels]), sum([(cur == -102).sum() for cur in new_labels]), new_labels.shape)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_user_input_embeds = [x[:tokenizer_model_max_length] for x in new_user_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
            new_informative_labels = [x[:tokenizer_model_max_length] for x in new_informative_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_user_input_embeds_padded = []
        new_labels_padded = torch.full(
            (batch_size, max_len),
            IGNORE_INDEX,
            dtype=new_labels[0].dtype,
            device=new_labels[0].device,
        )
        new_informative_labels_padded = torch.full(
            (batch_size, max_len),
            IGNORE_INDEX,
            dtype=new_informative_labels[0].dtype,
            device=new_informative_labels[0].device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device
        )
        position_ids = torch.zeros(
            (batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device
        )

        for i, (cur_new_embed, cur_user_new_embed, cur_new_labels, cur_new_informative_labels) in enumerate(zip(new_input_embeds, new_user_input_embeds, new_labels, new_informative_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                            cur_new_embed,
                        ),
                        dim=0,
                    )
                )
                new_user_input_embeds_padded.append(
                    torch.cat(
                        (
                            torch.zeros(
                                (max_len - cur_len, cur_user_new_embed.shape[1]),
                                dtype=cur_user_new_embed.dtype,
                                device=cur_user_new_embed.device,
                            ),
                            cur_user_new_embed,
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    new_informative_labels_padded[i, -cur_len:] = cur_new_informative_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )
            else:
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            cur_new_embed,
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                        ),
                        dim=0,
                    )
                )
                new_user_input_embeds_padded.append(
                    torch.cat(
                        (
                            cur_user_new_embed,
                            torch.zeros(
                                (max_len - cur_len, cur_user_new_embed.shape[1]),
                                dtype=cur_user_new_embed.dtype,
                                device=cur_user_new_embed.device,
                            ),
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    new_informative_labels_padded[i, :cur_len] = cur_new_informative_labels
                    attention_mask[i, :cur_len] = True
                    if shared_v_pid_stride is None:
                        position_ids[i, :cur_len] = torch.arange(
                            0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                        )
                    else:
                        cur_v_start_end = v_start_end[i]
                        cur_shared_position_ids = make_shared_position_ids(cur_v_start_end, cur_len, shared_v_pid_stride)
                        position_ids[i, :cur_len] = cur_shared_position_ids

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        new_user_input_embeds = torch.stack(new_user_input_embeds_padded, dim=0)
        if getattr(self.config, "combination") == "linear_combine":
            new_input_embeds = self.get_model().combination(torch.cat([new_input_embeds, new_user_input_embeds], dim=-1))
        elif getattr(self.config, "combination") == "adaptive_combine":
            adaptive_weight = self.get_model().combination(torch.cat([new_input_embeds, new_user_input_embeds], dim=-1))
            new_input_embeds = new_input_embeds * torch.sigmoid(adaptive_weight) + new_user_input_embeds * (1 - torch.sigmoid(adaptive_weight))
        elif getattr(self.config, "combination") == "add":
            new_input_embeds = (new_input_embeds + new_user_input_embeds) / 2
        else:
            raise NotImplementedError

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _informative_labels is None:
            informative_labels = None
        else:
            informative_labels = new_informative_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None and shared_v_pid_stride is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, informative_labels
    

    def prepare_inputs_labels_for_multimodal_inference(
        self, user_input_ids, position_ids, attention_mask, images, audios, video_audios, pad_token_id, padding_size
    ):

        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1).to(self.device) for x in image_features]
        else:
            image_features = self.encode_images(images).to(self.device)

        image_features = [e for e in image_features]

        audio_encoder = self.get_audio_encoder()
        if audios is not None:
            audio_features = audio_encoder(audios["audios"], audios["lengths"])
            state_labels = audios.get("state_labels", None)
            lengths_for_llm = audios["lengths_for_llm"]
            if state_labels is not None:
                assert len(audio_features["inputs_embeds"]) == len(state_labels) == len(lengths_for_llm)
        else:
            audio_features, state_labels, lengths_for_llm = None, None, None        

        if video_audios is not None:
            video_audio_features = audio_encoder(video_audios["audios"], video_audios["lengths"])
            video_audio_state_labels = video_audios.get("state_labels", None)
            video_audio_lengths_for_llm = video_audios["lengths_for_llm"]
            if video_audio_state_labels is not None:
                assert len(video_audio_features["inputs_embeds"]) == len(video_audio_state_labels) == len(video_audio_lengths_for_llm)
        else:
            video_audio_features, video_audio_state_labels, video_audio_lengths_for_llm = None, None, None  

        if attention_mask is None:
            attention_mask = torch.ones_like(user_input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(
                0, user_input_ids.shape[1], dtype=torch.long, device=user_input_ids.device
            )

        cur_image_idx = 0
        cur_audio_idx = 0
        cur_video_audio_idx = 0
        pooling_size = getattr(self.config, "pooling_size", 1)
        single_image_tokens = DEFAULT_IMAGE_TOKEN_NUMBER // pooling_size // pooling_size
        user_input_ids = [cur_input_ids for cur_input_ids in user_input_ids]
        new_user_input_embeds = []

        for batch_idx, cur_user_input_ids in enumerate(user_input_ids):

            num_images = (cur_user_input_ids == IMAGE_TOKEN_INDEX).sum() // single_image_tokens
            num_audio_frames = (cur_user_input_ids == AUDIO_TOKEN_INDEX).sum() // DEFAULT_AUDIO_TOKEN_NUMBER
            num_video_audio_frames = (cur_user_input_ids == VIDEO_AUDIO_TOKEN_INDEX).sum() // DEFAULT_AUDIO_TOKEN_NUMBER
            if num_images == 0 and num_audio_frames == 0 and num_video_audio_frames == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_audio_features = audio_features["inputs_embeds"][cur_audio_idx]
                cur_video_audio_features = video_audio_features["inputs_embeds"][cur_video_audio_idx]
                cur_user_input_embeds_1 = self.get_model().embed_tokens(cur_user_input_ids)
                cur_user_input_embeds = torch.cat(
                    [cur_user_input_embeds_1, cur_image_features[0:0], cur_audio_features[0:0], cur_video_audio_features[0:0]], dim=0
                )
                new_user_input_embeds.append(cur_user_input_embeds)
                cur_image_idx += 1
                cur_audio_idx += 1
                cur_video_audio_idx += 1
                continue


            multimodal_embedding_mask = cur_user_input_ids < 0
            multimodal_ids = cur_user_input_ids[multimodal_embedding_mask]
            cur_user_input_ids[multimodal_embedding_mask] = 0
            cur_user_input_embeds = self.get_model().embed_tokens(cur_user_input_ids)
            multimodal_embeddings = []
            point = 0
            while point < len(multimodal_ids):
                if multimodal_ids[point] == IMAGE_TOKEN_INDEX:
                    point += single_image_tokens
                    multimodal_embeddings.append(image_features[cur_image_idx])
                    cur_image_idx += 1
                elif multimodal_ids[point] == AUDIO_TOKEN_INDEX:
                    point += DEFAULT_AUDIO_TOKEN_NUMBER
                    multimodal_embeddings.append(audio_features["inputs_embeds"][cur_audio_idx])
                    cur_audio_idx += 1
                elif multimodal_ids[point] == VIDEO_AUDIO_TOKEN_INDEX:
                    point += DEFAULT_AUDIO_TOKEN_NUMBER
                    multimodal_embeddings.append(video_audio_features["inputs_embeds"][cur_video_audio_idx])
                    cur_video_audio_idx += 1
                else:
                    assert False, "Invalid multimodal token!"
            multimodal_embeddings = torch.cat(multimodal_embeddings)
            new_embeds = cur_user_input_embeds.clone()  # avoid in-place operator
            new_embeds[multimodal_embedding_mask] = multimodal_embeddings
            cur_user_new_input_embeds = [new_embeds]


            if num_audio_frames == 0:
                cur_audio_features = audio_features["inputs_embeds"][cur_audio_idx]
                cur_audio_idx += 1
                cur_user_new_input_embeds.append(cur_audio_features[0:0])
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_image_idx += 1
                cur_user_new_input_embeds.append(cur_image_features[0:0])
            if num_video_audio_frames == 0:
                cur_video_audio_features = video_audio_features["inputs_embeds"][cur_video_audio_idx]
                cur_video_audio_idx += 1
                cur_user_new_input_embeds.append(cur_video_audio_features[0:0])
            
            # for generation
            cur_user_new_input_embeds.append(self.get_model().embed_tokens(torch.full((padding_size,), pad_token_id, device=cur_user_input_ids.device, dtype=cur_user_input_ids.dtype)))
            cur_user_new_input_embeds = torch.cat(cur_user_new_input_embeds)

            new_user_input_embeds.append(cur_user_new_input_embeds)

        assert cur_image_idx == len(image_features)
        assert cur_audio_idx == audio_features["inputs_embeds"].shape[0]
        assert cur_video_audio_idx == video_audio_features["inputs_embeds"].shape[0]

        # Combine them
        max_len = max(x.shape[0] for x in new_user_input_embeds)
        batch_size = len(new_user_input_embeds)

        new_user_input_embeds_padded = []
        attention_mask = torch.zeros(
            (batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device
        )
        position_ids = torch.zeros(
            (batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device
        )

        for i, cur_user_new_embed in enumerate(new_user_input_embeds):
            cur_len = cur_user_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_user_input_embeds_padded.append(
                    torch.cat(
                        (
                            torch.zeros(
                                (max_len - cur_len, cur_user_new_embed.shape[1]),
                                dtype=cur_user_new_embed.dtype,
                                device=cur_user_new_embed.device,
                            ),
                            cur_user_new_embed,
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )
            else:
                new_user_input_embeds_padded.append(
                    torch.cat(
                        (
                            cur_user_new_embed,
                            torch.zeros(
                                (max_len - cur_len, cur_user_new_embed.shape[1]),
                                dtype=cur_user_new_embed.dtype,
                                device=cur_user_new_embed.device,
                            ),
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )

        new_user_input_embeds = torch.stack(new_user_input_embeds_padded, dim=0)

        return position_ids, attention_mask, new_user_input_embeds
    
    def initialize_vision_tokenizer(self, model_args, tokenizer):

        if model_args.use_new_seg_token:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_SEG_TOKEN, DEFAULT_BOS_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = True

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_combination_modules, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                lm_head_weight = mm_projector_weights['lm_head.weight']
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
                
                if output_embeddings.shape == lm_head_weight.shape:
                    output_embeddings[-num_new_tokens:] = lm_head_weight[-num_new_tokens:]
                elif lm_head_weight.shape[0] == num_new_tokens:
                    output_embeddings[-num_new_tokens:] = lm_head_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {lm_head_weight.shape}. Current: {lm_head_weight.shape}. Numer of new tokens: {num_new_tokens}.")



def merge_consecutive_tuples(tuples_list):
    if not tuples_list:
        return []

    # 首先对列表按照起点索引进行排序
    sorted_tuples = sorted(tuples_list, key=lambda x: x[0])
    
    # 初始化合并后的列表
    merged_tuples = [sorted_tuples[0]]
    
    for current_start, current_end in sorted_tuples[1:]:
        last_merged_start, last_merged_end = merged_tuples[-1]
        if current_start <= last_merged_end:  # 如果当前元组的起点小于等于上一个合并元组的终点
            # 合并这两个元组
            new_start, new_end = merged_tuples[-1][0], max(last_merged_end, current_end)
            merged_tuples[-1] = (new_start, new_end)
        else:
            # 如果当前元组不连续，直接添加到合并后的列表中
            merged_tuples.append((current_start, current_end))
    
    return merged_tuples


def make_shared_position_ids(cur_v_start_end, cur_len, shared_v_pid_stride):
    position_ids = torch.tensor([1.0] * cur_len)

    for start, end in cur_v_start_end:
        position_ids[start:end] = 1/shared_v_pid_stride
        v_mod = (end - start) % shared_v_pid_stride
        if v_mod != 0:
            position_ids[end-v_mod:end] = 1 / v_mod
    position_ids = position_ids.cumsum(dim=0)
    position_ids = torch.ceil(position_ids).long() - 1

    return position_ids

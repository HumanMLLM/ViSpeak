import argparse
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from utils.OVOBench import OVOBenchOffline
from decord import VideoReader, cpu
import math
from vispeak.model.builder import load_pretrained_model
from vispeak.util.mm_utils import get_model_name_from_path, tokenizer_image_audio_token, KeywordsStoppingCriteria
from vispeak.util.data_utils import _get_rawvideo_dec, SYSTEM_PROMTP
from vispeak.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_AUDIO_TOKEN,
    DEFAULT_SEG_TOKEN,
    MAX_IMAGE_LENGTH,
    DEFAULT_IMAGE_TOKEN_NUMBER,
    DEFAULT_AUDIO_TOKEN_NUMBER
)

class EvalViSpeak(OVOBenchOffline):
    def __init__(self, args):
        super().__init__(args)

        self.args = args
        self._model_init()
    
    def _model_init(self):
        model_path = self.args.model_path
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, processor, context_len = load_pretrained_model(
            model_path, None, model_name, 'qwen2p5_instruct'
        )
        self.tokenizer = tokenizer
        # model.bulid_informative_head('/mnt/data/shenghao/vita_v7/output/llava-s3-pretrain_video_informative/checkpoint-500/informative_head_params.bin')

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        image_processor = vision_tower.image_processor
        self.image_processor = image_processor

        audio_encoder = model.get_audio_encoder()
        audio_encoder.to(dtype=torch.float16)
        audio_processor = audio_encoder.audio_processor
        self.audio_processor = audio_processor
        model.eval()
        self.model = model
    
    
    def inference(self, video_file_name, prompt, start_time=0, end_time=0):
        try:
            pooling_size = getattr(self.model.config, "pooling_size", 1)
            patch_images, slice_len, audio_patch, all_num_audio_seg, sample_time = _get_rawvideo_dec(video_file_name, self.image_processor, max_frames=MAX_IMAGE_LENGTH*pooling_size*pooling_size, video_framerate=1, start_time=start_time, end_time=end_time, image_aspect_ratio=self.model.config.image_aspect_ratio)
            patch_images = torch.stack(patch_images).half().cuda()
            img_token_num = DEFAULT_IMAGE_TOKEN_NUMBER // pooling_size // pooling_size

            audios = dict()
            audio = torch.zeros(400, 80)
            audio_length = audio.shape[0]
            audio = torch.unsqueeze(audio, dim=0)
            audio_length = torch.unsqueeze(torch.tensor(audio_length), dim=0)
            audios['audios'] = audio.half().cuda()
            audios['lengths'] = audio_length.half().cuda()
            audio_for_llm_lens = 60
            audio_for_llm_lens = torch.unsqueeze(torch.tensor(audio_for_llm_lens), dim=0)
            audios["lengths_for_llm"] = audio_for_llm_lens.cuda()

            all_num_audio_seg = [0 for _ in all_num_audio_seg]
            audio_patch = []
            video_audios = dict()
            audio = torch.zeros(400, 80)
            audio_length = audio.shape[0]
            audio = torch.unsqueeze(audio, dim=0)
            audio_length = torch.unsqueeze(torch.tensor(audio_length), dim=0)
            video_audios['audios'] = audio.half().cuda()
            video_audios['lengths'] = audio_length.half().cuda()
            audio_for_llm_lens = 60
            audio_for_llm_lens = torch.unsqueeze(torch.tensor(audio_for_llm_lens), dim=0)
            video_audios["lengths_for_llm"] = audio_for_llm_lens.cuda()

            
            system_prompt = SYSTEM_PROMTP['video']
            print(prompt)

            user_input_timestamp = []
            for timestamp, num_video_audio in zip(sample_time, all_num_audio_seg):
                user_input_timestamp.append([timestamp, 'image'])
                for i in range(num_video_audio):
                    user_input_timestamp.append([timestamp + (i+1) * 0.1, 'video_audio']) # 0.1 is set randomly
            user_input_timestamp.append([end_time + 0.1, 'text'])

            # sort with timestamp
            user_input_timestamp.sort(key=lambda x: x[0])
            user_input_ids = []
            num_token = []
            
            # system prompt
            ids = tokenizer_image_audio_token(system_prompt + DEFAULT_SEG_TOKEN, self.tokenizer, return_tensors="pt")
            user_input_ids.append(ids)
            accumulated_token_num = len(ids)

            for timestamp in user_input_timestamp:
                if timestamp[1] == 'image':
                    ids = tokenizer_image_audio_token(DEFAULT_IMAGE_TOKEN + DEFAULT_SEG_TOKEN, self.tokenizer, return_tensors="pt", image_token_number=img_token_num)
                elif timestamp[1] == 'video_audio':
                    ids = tokenizer_image_audio_token(DEFAULT_VIDEO_AUDIO_TOKEN + DEFAULT_SEG_TOKEN, self.tokenizer, return_tensors="pt", image_token_number=img_token_num)
                else:
                    ids = tokenizer_image_audio_token(prompt + DEFAULT_SEG_TOKEN, self.tokenizer, return_tensors="pt", image_token_number=img_token_num)
                user_input_ids.append(ids)
                accumulated_token_num += len(ids)
                num_token.append(accumulated_token_num)
                
            user_input_ids = torch.cat(user_input_ids)

            start_inference_seg = [(i, num_token[i]) for i, timestamp in enumerate(user_input_timestamp) if timestamp[1] == 'text'][0]
            
            user_input_ids = user_input_ids.unsqueeze(0).cuda()

            keywords = ['<|im_end|>']
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, user_input_ids)

            cont, responce_time = self.model.streaming_generate(
                user_input_ids,
                start_inference_seg=start_inference_seg,
                timestamps=[t[0] for t in user_input_timestamp],
                seg_token_id=self.tokenizer.convert_tokens_to_ids(DEFAULT_SEG_TOKEN),
                images=patch_images,
                video_audios=video_audios,
                audios=audios,
                pad_token_id=self.tokenizer.pad_token_id,
                temperature=0.01,
                max_new_tokens=2048,
                stopping_criteria=stopping_criteria,
                proactive=False,
            )
            outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
            if '☞' in outputs or '☜' in outputs or '☟' in outputs:
                outputs = outputs[1:]

            print(outputs)
        except Exception as e:
            print(f'Error in dataset: {e}')
            outputs = ''
            responce_time = 0

        return outputs
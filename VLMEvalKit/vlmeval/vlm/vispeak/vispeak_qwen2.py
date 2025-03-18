import torch
from PIL import Image
from abc import abstractproperty
import sys
import os.path as osp
from ..base import BaseModel
from ...smp import *
from ...dataset import DATASET_TYPE
import copy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))
from vispeak.util.data_utils import dynamic_preprocess
from vispeak.constants import (
    DEFAULT_AUDIO_TOKEN,
    DEFAULT_DATA_RATIO,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    DEFAULT_VIDEO_AUDIO_TOKEN,
    DEFAULT_SEG_TOKEN,
    DEFAULT_IMAGE_TOKEN_NUMBER,
    IGNORE_INDEX,
    MAX_IMAGE_LENGTH,
    MIN_IMAGE_LENGTH,
)
import torchaudio
from vispeak.util.data_utils import preprocess_qwen2p5_moshi_offline_data, SYSTEM_PROMTP
from vispeak.util.mm_utils import tokenizer_image_audio_token
from dataclasses import dataclass, field
import transformers
from typing import Optional


class ModelArguments:
    model_name_or_path='/mnt/data/shenghao/models/VITA-1.5'
    vision_tower="/mnt/data/shenghao/models/InternViT-300M-448px"
    audio_encoder="/mnt/data/shenghao/models/VITA-1.5/audio-encoder-Qwen2-7B-1107-weight-base-11wh-tunning"
    pretrain_audio_mlp_adapter="output/llava-s1-pretrain_mlp/checkpoint-20317/audio_adpter.bin"
    pretrain_mm_mlp_adapter="output/llava-s1-pretrain_mlp/checkpoint-20317/mm_projector.bin"
    pretrain_combination_modules="output/llava-s1-pretrain_mlp/checkpoint-20317/combination_params.bin"
    mm_projector_type="mlp2x_gelu"
    combination="linear_combine" # ['linear_combine', 'add']



class ViSpeakQwen2(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = True

    DEFAULT_IMAGE_TOKEN = '<image>'
    IMAGE_TOKEN_INDEX = -200

    def __init__(self, model_path='/mnt/data/shenghao/models/VITA-1.5', **kwargs):
        assert model_path is not None
        try:
            from vispeak.model.builder import load_pretrained_model
            from vispeak.conversation import conv_templates
            from vispeak.util.mm_utils import get_model_name_from_path, tokenizer_image_token
        except:
            warnings.warn('Please install vita first.')

        model_args = ModelArguments()
        print(model_args)

        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None, model_name, model_type='qwen2p5_instruct', device_map='auto')
        #model.cuda().eval()
        # model.tie_weights()

        audio_encoder = model.get_audio_encoder()
        #audio_encoder.to(device="cuda", dtype=torch.float16)
        audio_encoder.to(dtype=torch.float16)
        audio_processor = audio_encoder.audio_processor

        # model.get_model().initialize_vision_modules(model_args=model_args)
        # model.get_model().initialize_combination_modules(model_args=model_args)
        # model.get_model().initialize_audio_modules(model_args=model_args)
        # model.cuda().to(dtype=torch.float16).eval()

        conv_mode = 'qwen2p5_instruct'
        self.stop_str = '<|im_end|>'
        self.conv_template = conv_mode
        self.conv_templates = conv_templates
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.tokenizer_image_token = tokenizer_image_token
        self.image_size = 448
        self.audio_processor = audio_processor

        self.use_video_audio = True

    def use_custom_prompt(self, dataset):
        return True

    def build_multi_choice_prompt(self, line, dataset=None):
        question = line['question']
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        if hint is not None:
            question = hint + '\n' + question

        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            question += f'\n{key}. {item}'
        prompt = question

        if len(options):
            prompt += '\n请直接回答选项字母。' if cn_string(
                prompt) else "\nAnswer with the option's letter from the given choices directly."
        else:
            prompt += '\n请直接回答问题。' if cn_string(prompt) else '\nAnswer the question directly.'

        return prompt

    def build_mme_rw_prompt(self, line, dataset_name):
        SYS = {
            'MME-RealWorld': (
                'Select the best answer to the above multiple-choice question based on the image. '
                'Respond with only the letter (A, B, C, D, or E) of the correct option. \n'
                'The best answer is:'
            ),
            'MME-RealWorld-CN': (
                '根据图像选择上述多项选择题的最佳答案。只需回答正确选项的字母（A, B, C, D 或 E）。\n'
                '最佳答案为：'
            ),
        }
        question = line['question']
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        if hint is not None:
            question = hint + '\n' + question

        choice_prompt = line['multi-choice options'] + '\n'
        question += ' ' + choice_prompt + SYS[dataset_name]

        prompt = question

        prompt += '\n请直接回答选项字母。' if cn_string(
            prompt) else "\nAnswer with the option's letter from the given choices directly."

        return prompt

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        if dataset is not None and listinstr(['MME'], dataset) and dataset != 'MME-RealWorld':
            question = line['question']
            prompt = question + ' Answer the question using a single word or phrase.'
        elif dataset is not None and listinstr(['HallusionBench'], dataset):
            question = line['question']
            prompt = question + ' Please answer yes or no. Answer the question using a single word or phrase.'
        elif dataset is not None and DATASET_TYPE(dataset) == 'MCQ' and dataset != 'MME-RealWorld':
            prompt = self.build_multi_choice_prompt(line, dataset)
        elif dataset == 'MME-RealWorld':
            prompt = self.build_mme_rw_prompt(line, dataset)
        elif dataset is not None and DATASET_TYPE(dataset) == 'VQA':
            if 'MathVista' in dataset:
                prompt = line['question']
                #prompt = 'According to the question shown in the image, please first conduct reasoning, and then answer the question and provide the final value, e.g., The answer is xxx\n' + line['question']
            elif listinstr(['LLaVABench'], dataset):
                question = line['question']
                prompt = question + '\nAnswer this question in detail.'
            elif listinstr(['MMVet'], dataset):
                prompt = line['question']
            else:
                question = line['question']
                prompt = question + '\nAnswer the question using a single word or phrase.'
        else:
            prompt = line['question']
        #message = [dict(type='text', value=prompt)]
        #message.extend([dict(type='image', value=s) for s in tgt_path])
        message = [dict(type='image', value=s) for s in tgt_path]
        message.extend([dict(type='text', value=prompt)])
        return message

    def set_max_num(self, dataset):
        if dataset is not None and listinstr(['ChartQA_TEST', 'MMMU_DEV_VAL'], dataset):
            self.max_num = 12
        elif dataset is not None and listinstr(['DocVQA_VAL', 'DocVQA_TEST'], dataset):
            self.max_num = 18
        elif dataset is not None and listinstr(['InfoVQA_VAL', 'InfoVQA_TEST', 'OCRBench'], dataset):
            self.max_num = 24
        elif dataset is not None and listinstr(['MVBench', 'Video-MME'], dataset):
            self.max_num = 1
        else:
            self.max_num = 6

    def generate_inner(self, message, dataset=None):
        from vispeak.util.mm_utils import KeywordsStoppingCriteria
        self.set_max_num(dataset)
        pooling_size = getattr(self.model.config, "pooling_size", 1)
        img_token_num = DEFAULT_IMAGE_TOKEN_NUMBER // pooling_size // pooling_size
        content, images, video_audios = '', [], []
        # print(message)
        user_input_timestamp = []
        user_input_ids = []
        num_token = []
        system_prompt = SYSTEM_PROMTP['video'] if listinstr(['MVBench', 'Video-MME'], dataset) else SYSTEM_PROMTP['image']
        ids = tokenizer_image_audio_token(system_prompt + DEFAULT_SEG_TOKEN, self.tokenizer, return_tensors="pt")
        user_input_ids.append(ids)
        accumulated_token_num = len(ids)
        if message[0]['type'] == 'text':
            message = message[1:]
        for i, msg in enumerate(message):
            if msg['type'] == 'text':
                if len(msg['value']) > 0:
                    ids = tokenizer_image_audio_token(msg['value'] + DEFAULT_SEG_TOKEN, self.tokenizer, return_tensors="pt", image_token_number=img_token_num)
                    user_input_ids.append(ids)
                    accumulated_token_num += len(ids)
                    num_token.append(accumulated_token_num)
                    user_input_timestamp.append([i, 'text'])
                    print(msg['value'])
            elif msg['type'] == 'image' and listinstr(['MVBench', 'Video-MME'], dataset):
                image = Image.open(msg['value']).convert('RGB')
                image, p_num = [image], [1]
                assert len(p_num) == 1
                #assert len(image) == p_num[0]
                images += image
                ids = tokenizer_image_audio_token(DEFAULT_IMAGE_TOKEN*p_num[0] + DEFAULT_SEG_TOKEN, self.tokenizer, return_tensors="pt", image_token_number=img_token_num)
                user_input_ids.append(ids)
                accumulated_token_num += len(ids)
                num_token.append(accumulated_token_num)
                user_input_timestamp.append([i, 'image'])
            elif msg['type'] == 'image':
                ## 这里分patch，同时计算patch数量
                image = Image.open(msg['value']).convert('RGB')
                image, p_num = dynamic_preprocess(image, min_num=1, max_num=self.max_num, image_size=self.image_size, use_thumbnail=True)
                assert len(p_num) == 1
                #assert len(image) == p_num[0]
                images += image
                ids = tokenizer_image_audio_token(DEFAULT_IMAGE_TOKEN*p_num[0] + DEFAULT_SEG_TOKEN, self.tokenizer, return_tensors="pt", image_token_number=img_token_num)
                user_input_ids.append(ids)
                accumulated_token_num += len(ids)
                num_token.append(accumulated_token_num)
                user_input_timestamp.append([i, 'image'])
            elif msg['type'] == 'video_audio' and self.use_video_audio:
                waveform, sample_rate = torchaudio.load(msg['value'])
                if sample_rate != 16000:
                    waveform = torchaudio.transforms.Resample(
                        orig_freq=sample_rate, new_freq=16000
                    )(waveform)
                video_audios.append(waveform)
                ids = tokenizer_image_audio_token(DEFAULT_VIDEO_AUDIO_TOKEN + DEFAULT_SEG_TOKEN, self.tokenizer, return_tensors="pt", image_token_number=img_token_num)
                user_input_ids.append(ids)
                accumulated_token_num += len(ids)
                num_token.append(accumulated_token_num)
                user_input_timestamp.append([i, 'video_audio'])
            # else:
            #     raise NotImplementedError

        preprocess = self.image_processor.preprocess
        image_tensor = [
            preprocess(f, return_tensors='pt')['pixel_values'][0].half().cuda() for f in images
        ]
        image_tensor = torch.stack(image_tensor)

        audio = []
        audio_for_llm_lens = []
        audio_length = []
        if len(video_audios) > 0:
            for seg_id in range(len(video_audios)):
                a, a_llm = self.audio_processor.process(video_audios[seg_id])
                audio.append(a)
                audio_for_llm_lens.append(a_llm)
                audio_length.append(a.shape[0])

        user_input_ids = torch.cat(user_input_ids)
        user_input_ids = user_input_ids.unsqueeze(0).cuda()
        start_inference_seg = [(i, num_token[i]) for i, timestamp in enumerate(user_input_timestamp) if timestamp[1] == 'text'][0]

        keywords = [self.stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, user_input_ids)

        video_audios = dict()
        if len(audio) == 0:
            audio = torch.zeros(400, 80)
            audio_length = audio.shape[0]
            audio = torch.unsqueeze(audio, dim=0)
            audio_length = torch.unsqueeze(torch.tensor(audio_length), dim=0)
            
            video_audios['audios'] = audio.half().cuda()
            video_audios['lengths'] = audio_length.half().cuda()
            audio_for_llm_lens = 60
            audio_for_llm_lens = torch.unsqueeze(torch.tensor(audio_for_llm_lens), dim=0)
            video_audios["lengths_for_llm"] = audio_for_llm_lens.cuda()
        else:
            video_audios['audios'] = torch.stack(audio).half().cuda()
            video_audios['lengths'] = torch.tensor(audio_length).half().cuda()
            video_audios["lengths_for_llm"] = torch.tensor(audio_for_llm_lens).cuda()
        
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
        
        cont, responce_time = self.model.streaming_generate(
            user_input_ids,
            start_inference_seg=start_inference_seg,
            timestamps=[t[0] for t in user_input_timestamp],
            seg_token_id=self.tokenizer.convert_tokens_to_ids(DEFAULT_SEG_TOKEN),
            images=image_tensor,
            video_audios=video_audios,
            audios=audios,
            pad_token_id=self.tokenizer.pad_token_id,
            temperature=0.01,
            max_new_tokens=2048,
            stopping_criteria=stopping_criteria,
        )

        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
        if '☞' in text_outputs:
            text_outputs = text_outputs.split('☞')[-1]
            return text_outputs
        elif '☜' in text_outputs:
            text_outputs = text_outputs.split('☜')[-1]
            return text_outputs
        elif '☟' in text_outputs:
            text_outputs = text_outputs.split('☟')[-1]
            return text_outputs
        else:
            return text_outputs



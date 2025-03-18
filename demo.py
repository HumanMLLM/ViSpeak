
import argparse
import json
import os
import random
import torch
from tqdm import tqdm

from vispeak.constants import (
    DEFAULT_SEG_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    MAX_IMAGE_LENGTH,
    DEFAULT_IMAGE_TOKEN_NUMBER,
)

from vispeak.model.builder import load_pretrained_model
from vispeak.util.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    tokenizer_image_audio_token,
)
from vispeak.util.data_utils import _get_rawvideo_dec, SYSTEM_PROMTP



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--video", type=str, required=True)
    args = parser.parse_args()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(
        args.model_path, None, model_name, 'qwen2p5_instruct'
    )

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    image_processor = vision_tower.image_processor

    audio_encoder = model.get_audio_encoder()
    audio_encoder.to(dtype=torch.float16)
    audio_processor = audio_encoder.audio_processor
    model.eval()

    pooling_size = getattr(model.config, "pooling_size", 1)
    patch_images, slice_len, audio_patch, all_num_audio_seg, sample_time = _get_rawvideo_dec(args.video, image_processor, max_frames=MAX_IMAGE_LENGTH*pooling_size*pooling_size, video_framerate=1, image_aspect_ratio=model.config.image_aspect_ratio)
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


    system_prompt = SYSTEM_PROMTP['vispeak']

    user_input_timestamp = []
    for timestamp in sample_time:
        user_input_timestamp.append([timestamp, 'image'])

    # sort with timestamp
    user_input_timestamp.sort(key=lambda x: x[0])
    user_input_ids = []
    num_token = []
    
    # system prompt
    ids = tokenizer_image_audio_token(system_prompt + DEFAULT_SEG_TOKEN, tokenizer, return_tensors="pt")
    user_input_ids.append(ids)
    accumulated_token_num = len(ids)
    system_prompt_num = len(ids)

    text_id = 0
    for timestamp in user_input_timestamp:
        ids = tokenizer_image_audio_token(DEFAULT_IMAGE_TOKEN + DEFAULT_SEG_TOKEN, tokenizer, return_tensors="pt", image_token_number=img_token_num)
        user_input_ids.append(ids)
        accumulated_token_num += len(ids)
        num_token.append(accumulated_token_num)
        
    user_input_ids = torch.cat(user_input_ids)
    agent_input_ids = torch.full((len(user_input_ids),), tokenizer.pad_token_id, device=user_input_ids.device, dtype=user_input_ids.dtype)

    start_inference_seg = (0, num_token[0])
    user_input_ids = user_input_ids.unsqueeze(0).cuda()
    agent_input_ids = agent_input_ids.unsqueeze(0).cuda()

    keywords = ['<|im_end|>']
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, user_input_ids)

    cont, responce_time = model.streaming_generate(
        user_input_ids,
        agent_input_ids=agent_input_ids,
        start_inference_seg=start_inference_seg,
        timestamps=[t[0] for t in user_input_timestamp],
        seg_token_id=tokenizer.convert_tokens_to_ids(DEFAULT_SEG_TOKEN),
        images=patch_images,
        video_audios=video_audios,
        audios=audios,
        pad_token_id=tokenizer.pad_token_id,
        temperature=0.01,
        max_new_tokens=2048,
        padding_size=128,
        stopping_criteria=stopping_criteria,
        proactive=True,
        sentence_end_token_id=tokenizer.convert_tokens_to_ids('<|im_end|>'),
    )
    outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
    if '☞' in outputs or '☜' in outputs or '☟' in outputs:
        outputs = outputs[1:]
    print(outputs, responce_time)


"""
export PYTHONPATH=./

python dmeo.py --model_path /mnt/data/qize.yqz/pretrained_models/VITA-MLLM/VITA-1.5/ --video demo.mp4

"""

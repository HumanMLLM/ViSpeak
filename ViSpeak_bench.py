
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



ViSpeak_Bench_metainfo = {
    'Anomaly_Warning': 'annotations/Anomaly_Warning.json',
    'Gesture_Understanding': 'annotations/Gesture_Understanding_v3.json',
    'Humor_Reaction': 'annotations/Humor_Reaction.json',
    'Visual_Interruption': 'annotations/Visual_Interruption.json',
    'Visual_Reference': 'annotations/Visual_Reference.json',
    'Visual_Termination': 'annotations/Visual_Termination.json',
    'Visual_Wake-Up': 'annotations/Visual_Wake-Up.json',
}


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--video_folder_path", type=str, required=True)
    args = parser.parse_args()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(
        args.model_path, None, model_name, 'qwen2p5_instruct'
    )
    # model.bulid_informative_head('/mnt/data/shenghao/vita_v7/output/llava-s3-pretrain_video_informative/checkpoint-500/informative_head_params.bin')

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    image_processor = vision_tower.image_processor

    audio_encoder = model.get_audio_encoder()
    audio_encoder.to(dtype=torch.float16)
    audio_processor = audio_encoder.audio_processor
    model.eval()

    for task in ViSpeak_Bench_metainfo.keys():
        with open(os.path.join(args.video_folder_path, ViSpeak_Bench_metainfo[task])) as f:
            data = json.load(f)
        answers = []
        if os.path.exists(f"{task}_output.json"):
            continue

        for da in tqdm(data):
            # print(da)
            video_path = os.path.join(args.video_folder_path, da['video'])
            video_start_time = da.get("video_start_time", 0)
            video_end_time = da.get("video_end_time", None)

            pooling_size = getattr(model.config, "pooling_size", 1)
            patch_images, slice_len, audio_patch, all_num_audio_seg, sample_time = _get_rawvideo_dec(video_path, image_processor, max_frames=MAX_IMAGE_LENGTH*pooling_size*pooling_size, video_framerate=1, start_time=video_start_time, end_time=video_end_time, image_aspect_ratio=model.config.image_aspect_ratio)
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


            system_prompt = SYSTEM_PROMTP['vispeak'] if task != 'Visual_Reference' else SYSTEM_PROMTP['video']

            user_input_timestamp = []
            for timestamp in sample_time:
                user_input_timestamp.append([timestamp, 'image'])
            
            text_input = []
            if task != 'Visual_Reference':
                for conv in da['conversations']:
                    if conv['from'] == 'human':
                        text_input.append(conv['value'])
                        user_input_timestamp.append([conv['time'], 'text'])
            else:
                text_input.append(da['questions'])
                user_input_timestamp.append([100000, 'text'])

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
                if timestamp[1] == 'image':
                    ids = tokenizer_image_audio_token(DEFAULT_IMAGE_TOKEN + DEFAULT_SEG_TOKEN, tokenizer, return_tensors="pt", image_token_number=img_token_num)
                else:
                    ids = tokenizer_image_audio_token(text_input[text_id] + DEFAULT_SEG_TOKEN, tokenizer, return_tensors="pt", image_token_number=img_token_num)
                    text_id += 1
                user_input_ids.append(ids)
                accumulated_token_num += len(ids)
                num_token.append(accumulated_token_num)
                
            user_input_ids = torch.cat(user_input_ids)
            agent_input_ids = torch.full((len(user_input_ids),), tokenizer.pad_token_id, device=user_input_ids.device, dtype=user_input_ids.dtype)

            if task == 'Anomaly_Warning' or task == 'Humor_Reaction' or task == 'Visual_Wake-Up':
                start_inference_seg = (0, num_token[0])
            # elif task == 'Gesture_Understanding':
            #     ids = tokenizer_image_audio_token("☜" + da['conversations'][0]['value'] + "<|im_end|>", tokenizer, return_tensors="pt", image_token_number=img_token_num)
            #     agent_input_ids[num_token[0]: num_token[0] + len(ids)] = ids
            #     for j in range(len(num_token)):
            #         if num_token[j] > num_token[0] + len(ids):
            #             break
            #     start_inference_seg = (j, num_token[j])
            elif task == 'Visual_Interruption':
                ids = tokenizer_image_audio_token("☜" + da['conversations'][1]['value'] + "<|im_end|>", tokenizer, return_tensors="pt", image_token_number=img_token_num)
                start_token = system_prompt_num
                for j, (timestamp, num) in enumerate(zip(user_input_timestamp, num_token)):
                    if timestamp[1] == 'text':
                        start_token = num
                        break
                ids_len = len(ids)
                if start_token + len(ids) >= len(agent_input_ids):
                    ids_len = len(agent_input_ids) - start_token
                agent_input_ids[start_token: start_token + ids_len] = ids[:ids_len]
                start_inference_seg = (j + 1, num_token[j + 1])
            elif task == 'Visual_Reference':
                start_inference_seg = (len(num_token) - 1, num_token[-1])
            elif task == 'Visual_Termination' or task == 'Gesture_Understanding':
                ids = tokenizer_image_audio_token("☜" + da['conversations'][1]['value'] + "<|im_end|>", tokenizer, return_tensors="pt", image_token_number=img_token_num)
                start_token = system_prompt_num
                for j, (timestamp, num) in enumerate(zip(user_input_timestamp, num_token)):
                    if timestamp[1] == 'text':
                        start_token = num
                        break
                ids_len = len(ids)
                if start_token + len(ids) >= len(agent_input_ids):
                    ids_len = len(agent_input_ids) - start_token
                agent_input_ids[start_token: start_token + ids_len] = ids[:ids_len]
                for j in range(len(num_token)):
                    if num_token[j] > start_token + ids_len:
                        break
                start_inference_seg = (j, num_token[j])
            else:
                raise NotImplementedError
            
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
                proactive=True if task != 'Visual_Reference' else False,
                sentence_end_token_id=tokenizer.convert_tokens_to_ids('<|im_end|>'),
                interrupt=True if task == 'Visual_Interruption' else False,
            )
            outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
            if '☞' in outputs or '☜' in outputs or '☟' in outputs:
                outputs = outputs[1:]
            print(outputs, responce_time)
            answers.append({da['video']: [outputs, responce_time]})

        with open(f"{task}_output.json", 'w') as f:
            json.dump(answers, f, indent=4)

        
    with open(os.path.join(args.video_folder_path, ViSpeak_Bench_metainfo['Visual_Reference'])) as f:
        answers = json.load(f)
    with open("Visual_Reference_output.json") as f:
        responces = json.load(f)

    num_correct = 0
    for responce, answer in zip(responces, answers):
        video_name = list(responce.keys())[0]
        assert video_name == answer['video'], "The order of responces do not align with the question file"
        output = responce[video_name][0].lower()
        if answer['ans'].lower() == output[0]:
            num_correct += 1
        elif answer['ans_full'] in output:
            num_correct += 1
        
    print(f"Total number of qusetions in Visual Reference is {len(answers)}")
    print(f"Total number of qusetions with correct answers is {num_correct}")
    print(f"Accuracy: {num_correct / len(answers)}")



"""
export PYTHONPATH=./

python -m torch.distributed.launch --use_env  --nproc_per_node 1 --nnodes 1 vita/audio_eval/eval_asr.py \
   --model_path /mnt/data/qize.yqz/pretrained_models/VITA-MLLM/VITA-1.5/ 

"""

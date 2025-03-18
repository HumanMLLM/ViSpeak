import copy
import json
import math
import os
import random
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import re

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import transformers
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from decord import VideoReader, cpu, AudioReader
from vispeak import conversation as conversation_lib
from vispeak.config import DataConfig
from vispeak.constants import (
    DEFAULT_AUDIO_TOKEN,
    DEFAULT_DATA_RATIO,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    DEFAULT_VIDEO_AUDIO_TOKEN,
    DEFAULT_SEG_TOKEN,
    IGNORE_INDEX,
    MAX_IMAGE_LENGTH,
    MIN_IMAGE_LENGTH,
    DEFAULT_AUDIO_TOKEN_NUMBER,
    DEFAULT_IMAGE_TOKEN_NUMBER,
    IMAGE_TOKEN_INDEX,
    AUDIO_TOKEN_INDEX,
    VIDEO_AUDIO_TOKEN_INDEX,
)
from vispeak.util.mm_utils import tokenizer_image_audio_token, tokenizer_image_token


@dataclass
class DataArguments:
    lazy_preprocess: bool = False
    is_multimodal: bool = True
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = field(default=None)
    dataset_use: str = field(default="temp")
    min_dynamic_patch: int = 1
    max_dynamic_patch: int = 12
    use_thumbnail: bool = True
    audio_segment_len: float = 1.0 # 1s
    max_video_audio_segment: int = 6
    video_framerate: int = 1
    pooling_size: int = 1
    minimal_image_ratio: int = 1


OFFLINE_IMAGE_PROMPT = "You are an AI robot and now processing offline data. You should first read all input information and then answer questions. Please note that you are seeing the image, not the video."
OFFLINE_VIDEO_PROMPT = "You are an AI robot and now processing offline data. You should first read all input information and then answer questions. Please note that you are seeing the video, not the image."
OFFLINE_AUDIO_PROMPT = "You are an AI robot and now processing offline data. You should first read all input information and then answer questions. Please note that you are listening some audios."
OFFLINE_TEXT_PROMPT = "You are an AI robot and now processing offline data. You should first read all input information and then answer questions. Please note that you are reading texts."
ONLINE_PROMPT = "You are an AI robot and now processing online videos. Users will ask any questions at any time and you should first determine when to answer the questions and then answer the questions based on what you have seen."
ViSpeak_PROMPT = "You are an AI robot and now processing online videos. You should be aware of anomaly events, fun things, and users actions and gestures to provide in-time interaction with users and necessary assistance effectively."

SYSTEM_PROMTP = {
    'image': OFFLINE_IMAGE_PROMPT,
    'video': OFFLINE_VIDEO_PROMPT,
    'audio': OFFLINE_AUDIO_PROMPT,
    'text': OFFLINE_TEXT_PROMPT,
    'online': ONLINE_PROMPT,
    'vispeak': ViSpeak_PROMPT,
}


def preprocess_multimodal_offline(
    sources: Sequence[str],
    data_args: DataArguments,
    patch_num=[1],
    audio_segment_num=[1],
    video_audio_segment_num=[0], # 这里假设一条数据只有一个视频
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    k_img_ph = 0
    k_audio_ph = 0
    for source in sources:
        multimodal_type = 'text'
        for i, sentence in enumerate(source):
            if DEFAULT_IMAGE_TOKEN in sentence["value"] or DEFAULT_VIDEO_TOKEN in sentence["value"] or DEFAULT_AUDIO_TOKEN in sentence["value"]:
                # streaming data do not needs '\n' to seperate different input modality
                # instead we use <seg> token to seperate different input modality explicitly
                sentence["value"] = ( # delete "\n"
                    sentence["value"]
                    .replace(DEFAULT_IMAGE_TOKEN + "\n", DEFAULT_IMAGE_TOKEN)
                    .strip()
                )
                sentence["value"] = ( # delete "\n"
                    sentence["value"]
                    .replace("\n" + DEFAULT_IMAGE_TOKEN, DEFAULT_IMAGE_TOKEN)
                    .strip()
                )
                sentence["value"] = ( # delete "\n"
                    sentence["value"]
                    .replace(DEFAULT_VIDEO_TOKEN + "\n", DEFAULT_VIDEO_TOKEN)
                    .strip()
                )
                sentence["value"] = ( # delete "\n"
                    sentence["value"]
                    .replace("\n" + DEFAULT_VIDEO_TOKEN, DEFAULT_VIDEO_TOKEN)
                    .strip()
                )
                sentence["value"] = ( # delete "\n"
                    sentence["value"]
                    .replace(DEFAULT_AUDIO_TOKEN + "\n", DEFAULT_AUDIO_TOKEN)
                    .strip()
                )
                sentence["value"] = ( # delete "\n"
                    sentence["value"]
                    .replace("\n" + DEFAULT_AUDIO_TOKEN, DEFAULT_AUDIO_TOKEN)
                    .strip()
                )
                # make sure that the prompt order is <image>question
                if sentence["value"].endswith(DEFAULT_IMAGE_TOKEN):
                    IMAGE_TOKEN_NUM = sentence["value"].count(DEFAULT_IMAGE_TOKEN)
                    sentence["value"] = (
                        sentence["value"].replace(DEFAULT_IMAGE_TOKEN * IMAGE_TOKEN_NUM, "").strip()
                    )
                    sentence["value"] = DEFAULT_IMAGE_TOKEN * IMAGE_TOKEN_NUM + sentence["value"]
                    sentence["value"] = sentence["value"].strip()
                # make sure that the prompt order is <video>question
                if sentence["value"].endswith(DEFAULT_VIDEO_TOKEN):
                    VIDEO_TOKEN_NUM = sentence["value"].count(DEFAULT_VIDEO_TOKEN)
                    sentence["value"] = (
                        sentence["value"].replace(DEFAULT_VIDEO_TOKEN * VIDEO_TOKEN_NUM, "").strip()
                    )
                    sentence["value"] = DEFAULT_VIDEO_TOKEN * VIDEO_TOKEN_NUM + sentence["value"]
                    sentence["value"] = sentence["value"].strip()

                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence["value"] = sentence["value"].replace(
                        DEFAULT_IMAGE_TOKEN, "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>"
                    )

            # text also needs a seg token
            new_prompt = []
            if sentence["from"] == 'human':
                for chunk in re.split(r"(<audio>|<image>|<video>)", sentence["value"]):
                    if chunk == "<audio>":
                        new_prompt.append(DEFAULT_AUDIO_TOKEN)
                    elif chunk == "<image>":
                        new_prompt.append(DEFAULT_IMAGE_TOKEN)
                    elif chunk == "<video>":
                        new_prompt.append(DEFAULT_VIDEO_TOKEN)
                    elif len(chunk) > 0:
                        new_prompt.append(chunk + DEFAULT_SEG_TOKEN)
                sentence["value"] = "".join(new_prompt)

            # AnyRes for images
            if DEFAULT_IMAGE_TOKEN in sentence["value"]:
                multimodal_type = 'image'
                replace_token = DEFAULT_IMAGE_TOKEN * patch_num[k_img_ph] + DEFAULT_SEG_TOKEN # we select a rarely-used token as seg token
                k_img_ph += 1
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

            # multi frames in vides
            vid_replace_token = ''
            if DEFAULT_VIDEO_TOKEN in sentence["value"]:
                multimodal_type = 'video'
                for audio_num in video_audio_segment_num:
                    vid_replace_token += DEFAULT_IMAGE_TOKEN + DEFAULT_SEG_TOKEN + (DEFAULT_VIDEO_AUDIO_TOKEN + DEFAULT_SEG_TOKEN) * audio_num
                sentence["value"] = sentence["value"].replace(DEFAULT_VIDEO_TOKEN, vid_replace_token)

            # multi segments for audios
            if DEFAULT_AUDIO_TOKEN in sentence["value"]:
                if multimodal_type == 'text':
                    multimodal_type = 'audio'
                audio_replace_token = (DEFAULT_AUDIO_TOKEN + DEFAULT_SEG_TOKEN) * audio_segment_num[k_audio_ph]
                k_audio_ph += 1
                sentence["value"] = sentence["value"].replace(DEFAULT_AUDIO_TOKEN, audio_replace_token)

            # respond needs a start token and an end token
            # we reuse the token from VITA
            # ☞ for audio question
            # ☜ for text question
            # ☟ for visual proactive output
            if sentence["from"] == 'gpt':
                if source[i - 1]["value"].endswith(DEFAULT_AUDIO_TOKEN): # audio question
                    sentence["value"] = "☞" + sentence["value"] + "<|im_end|>"
                else: # text question
                    sentence["value"] = "☜" + sentence["value"] + "<|im_end|>"

            sentence["value"] = sentence["value"].replace("\n\n", "\n")
        source.insert(0, {"from": "human", "value": SYSTEM_PROMTP[multimodal_type] + DEFAULT_SEG_TOKEN})
    return sources



# moshi-like template
# do not use role control any more
def preprocess_qwen2p5_moshi_offline_data(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    img_token_num,
) -> Dict:
    # Apply prompt templates
    all_user_input_ids = []
    all_agent_input_ids = []
    all_targets = []
    for i, source in enumerate(sources):
        user_input_ids = []
        agent_input_ids = []
        targets = []
        if source[0]["from"] != 'human':
            # Skip the first one if it is not from human
            source = source[1:]

        for j, sentence in enumerate(source):
            if sentence['from'] == 'human':
                user_input_id = tokenizer_image_audio_token(sentence['value'], tokenizer, return_tensors="pt", image_token_number=img_token_num)
                user_input_ids.append(user_input_id)
                agent_input_id = torch.tensor([tokenizer.pad_token_id for _ in range(len(user_input_id))], dtype=torch.long)
                agent_input_ids.append(agent_input_id)
                targets.append(torch.tensor([IGNORE_INDEX for _ in range(len(user_input_id))], dtype=torch.long))
            else: # sentence[0]['from'] == 'gpt'
                agent_input_id = tokenizer_image_audio_token(sentence['value'], tokenizer, return_tensors="pt", image_token_number=img_token_num)
                agent_input_ids.append(agent_input_id)
                user_input_id = torch.tensor([tokenizer.pad_token_id for _ in range(len(agent_input_id))], dtype=torch.long)
                user_input_ids.append(user_input_id)
                targets.append(agent_input_id)
        
        user_input_ids = torch.cat(user_input_ids)
        agent_input_ids = torch.cat(agent_input_ids)
        targets = torch.cat(targets)
        # seg token needs supervision
        # next_seg_mask = torch.cat([user_input_ids[:1], user_input_ids[:-1]]) == tokenizer.convert_tokens_to_ids(DEFAULT_SEG_TOKEN)
        # targets[(targets == IGNORE_INDEX) & next_seg_mask] = tokenizer.pad_token_id
        assert len(user_input_ids) == len(agent_input_ids)
        all_user_input_ids.append(user_input_ids)
        all_agent_input_ids.append(agent_input_ids)
        all_targets.append(targets)

    all_user_input_ids = torch.stack(all_user_input_ids)
    all_agent_input_ids = torch.stack(all_agent_input_ids)
    all_targets = torch.stack(all_targets)
    informative_labels = torch.zeros_like(all_targets, device=all_targets.device, dtype=all_targets.dtype)
    informative_labels[:] = -100

    return dict(
        all_user_input_ids=all_user_input_ids,
        all_agent_input_ids=all_agent_input_ids,
        labels=all_targets,
        informative_labels=informative_labels,
    )




# moshi-like template
# do not use role control any more
# The used data must be video and the questions must be in text
# conversations can start with gpt
def preprocess_qwen2p5_moshi_online_data(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    img_token_num,
    sample_time,
    proactive_flags,
    video_audio_segment_num=[0], # 这里假设一条数据只有一个视频
) -> Dict:
    # Apply prompt templates
    all_user_input_ids = []
    all_agent_input_ids = []
    all_targets = []
    all_informative_labels = []
    for j, (source, flag) in enumerate(zip(sources, proactive_flags)):
        user_input_ids = []

        user_input_timestamp = []
        for timestamp, num_video_audio in zip(sample_time, video_audio_segment_num):
            user_input_timestamp.append([timestamp, 'image'])
            for i in range(num_video_audio):
                user_input_timestamp.append([timestamp + (i+1) * 0.1, 'video_audio']) # 0.1 is set randomly

        user_conversations = []
        has_text = False
        for sentence in source:
            if sentence['from'] == 'human':
                has_text = True
                user_conversations.append(sentence['value'])
                user_input_timestamp.append([sentence['time'], 'text'])
        
        # sort with timestamp
        user_input_timestamp.sort(key=lambda x: x[0])

        # system prompt
        sys_prompt = SYSTEM_PROMTP["online"] if has_text else SYSTEM_PROMTP["vispeak"]
        ids = tokenizer_image_audio_token(sys_prompt + DEFAULT_SEG_TOKEN, tokenizer, return_tensors="pt")
        user_input_ids.append(ids)
        system_prompt_len = len(ids)

        num_text = 0
        num_token = []
        accumulated_token_num = system_prompt_len
        for timestamp in user_input_timestamp:
            if timestamp[1] == 'image':
                ids = tokenizer_image_audio_token(DEFAULT_IMAGE_TOKEN + DEFAULT_SEG_TOKEN, tokenizer, return_tensors="pt", image_token_number=img_token_num)
            elif timestamp[1] == 'video_audio':
                ids = tokenizer_image_audio_token(DEFAULT_VIDEO_AUDIO_TOKEN + DEFAULT_SEG_TOKEN, tokenizer, return_tensors="pt", image_token_number=img_token_num)
            else:
                ids = tokenizer_image_audio_token(user_conversations[num_text] + DEFAULT_SEG_TOKEN, tokenizer, return_tensors="pt", image_token_number=img_token_num)
                num_text += 1
            user_input_ids.append(ids)
            accumulated_token_num += len(ids)
            num_token.append(accumulated_token_num)
        
        max_token_num = accumulated_token_num
        preserved_num_token = copy.deepcopy(num_token)
        user_input_ids = torch.cat(user_input_ids)
        agent_input_ids = torch.full((len(user_input_ids)+512,), tokenizer.pad_token_id, device=user_input_ids.device, dtype=user_input_ids.dtype)
        targets = torch.full((len(user_input_ids)+512,), IGNORE_INDEX, device=user_input_ids.device, dtype=user_input_ids.dtype)
        informative_label = torch.full((len(user_input_ids)+512,), IGNORE_INDEX, device=user_input_ids.device, dtype=user_input_ids.dtype)
        total_len = len(user_input_ids)
        
        gpt_conversations = []
        for sentence in source:
            if sentence['from'] == 'gpt':
                gpt_conversations.append(sentence)
        
        if not flag:
            # Streaming数据的answer一定跟在question的text segment后面
            num_token = [num_token[i] for i, timestamp in enumerate(user_input_timestamp) if timestamp[1] == 'text']
            insert_timestamps = [timestamp[0] for i, timestamp in enumerate(user_input_timestamp) if timestamp[1] == 'text']
            num_token = [system_prompt_len] + num_token # 错开了一位,如果时间比当前question早，就跟在上一个question后面
            insert_timestamps = insert_timestamps + [1000000] # prevent overflow
            start_token = "☜"
        else:
            # Proactive数据的answer一定跟在image segment后面，并且时间不早于question的时间，每条数据最多仅有一个question
            query_time = [timestamp[0] for i, timestamp in enumerate(user_input_timestamp) if timestamp[1] == 'text']
            if len(query_time) > 0:
                query_time = query_time[0]
            else:
                query_time = -1
            num_token = [num_token[i] for i, timestamp in enumerate(user_input_timestamp) if timestamp[1] == 'image' and timestamp[0] > query_time]
            insert_timestamps = [timestamp[0] for i, timestamp in enumerate(user_input_timestamp) if timestamp[1] == 'image' and timestamp[0] > query_time]
            num_token = [system_prompt_len] + num_token # 错开了一位,如果时间比当前image早，就跟在上一个image后面
            insert_timestamps = insert_timestamps + [1000000] # prevent overflow
            start_token = "☟"

        user_time_id = 0
        for sentence in gpt_conversations:
            while sentence['time'] >= insert_timestamps[user_time_id]:
                user_time_id += 1
            # if sentence['time'] - start_time < insert_timestamps[user_time_id]
            ids = tokenizer_image_audio_token(start_token + sentence['value'] + "<|im_end|>", tokenizer, return_tensors="pt")
            max_token_num = max(max_token_num, num_token[user_time_id] + len(ids))
            agent_input_ids[num_token[user_time_id]: num_token[user_time_id] + len(ids)] = ids
            targets[num_token[user_time_id]: num_token[user_time_id] + len(ids)] = ids

        # proactive data should apply supervision on informative head
        if flag:
            visual_seg = [preserved_num_token[i] - 2 for i, timestamp in enumerate(user_input_timestamp) if timestamp[1] == 'image']
            for token_id in visual_seg:
                informative_label[token_id] = 0
            for sentence in source:
                if sentence['from'] == 'gpt':
                    query_time = [timestamp[0] for i, timestamp in enumerate(user_input_timestamp) if timestamp[1] == 'text']
                    if len(query_time) > 0:
                        query_time = query_time[0]
                    else:
                        query_time = -1
                    begin_time = (sentence['timespan'][0] + sentence['timespan'][1]) / 2
                    pos_token = [preserved_num_token[i] - 2 for i, timestamp in enumerate(user_input_timestamp) if timestamp[1] == 'image' and timestamp[0] > query_time and timestamp[0] >= begin_time and timestamp[0] <= sentence['timespan'][1]]
                    for token_id in pos_token:
                        informative_label[token_id] = 1

        agent_input_ids = agent_input_ids[:max_token_num]
        targets = targets[:max_token_num]
        informative_label = informative_label[:max_token_num]
        if max_token_num > len(user_input_ids):
            user_input_ids = torch.cat([user_input_ids, torch.full((max_token_num - len(user_input_ids),), tokenizer.pad_token_id, device=user_input_ids.device, dtype=user_input_ids.dtype)])
        
        assert len(user_input_ids) == len(agent_input_ids)
        all_user_input_ids.append(user_input_ids)
        all_agent_input_ids.append(agent_input_ids)
        all_targets.append(targets)
        all_informative_labels.append(informative_label)

    all_user_input_ids = torch.stack(all_user_input_ids)
    all_agent_input_ids = torch.stack(all_agent_input_ids)
    all_targets = torch.stack(all_targets)
    informative_labels = torch.stack(all_informative_labels)

    return dict(
        all_user_input_ids=all_user_input_ids,
        all_agent_input_ids=all_agent_input_ids,
        labels=all_targets,
        informative_labels=informative_labels,
    )



# moshi-like template
# do not use role control any more
# The used data must be video and the questions must be in text
# conversations can start with gpt
def preprocess_qwen2p5_moshi_audio_proactive_data(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    img_token_num,
    sample_time,
    use_audios,
    no_visual_proactive,
    video_audio_segment_num=[0], # 这里假设一条数据只有一个视频
) -> Dict:
    # Apply prompt templates
    all_user_input_ids = []
    all_agent_input_ids = []
    all_targets = []
    all_informative_labels = []
    for j, (source, flag) in enumerate(zip(sources, use_audios)):
        user_input_ids = []

        user_input_timestamp = []
        for timestamp, num_video_audio in zip(sample_time, video_audio_segment_num):
            user_input_timestamp.append([timestamp, 'image'])
            if flag:
                for i in range(num_video_audio):
                    user_input_timestamp.append([timestamp + (i+1) * 0.1, 'video_audio']) # 0.1 is set randomly

        user_conversations = []
        if not flag:
            for sentence in source:
                if sentence['from'] == 'human':
                    user_conversations.append(sentence['value'])
                    user_input_timestamp.append([sentence['time'], 'text'])
        
        # sort with timestamp
        user_input_timestamp.sort(key=lambda x: x[0])

        # system prompt
        ids = tokenizer_image_audio_token(SYSTEM_PROMTP["vispeak"] + DEFAULT_SEG_TOKEN, tokenizer, return_tensors="pt")
        user_input_ids.append(ids)
        system_prompt_len = len(ids)

        num_text = 0
        num_token = []
        accumulated_token_num = system_prompt_len
        for timestamp in user_input_timestamp:
            if timestamp[1] == 'image':
                ids = tokenizer_image_audio_token(DEFAULT_IMAGE_TOKEN + DEFAULT_SEG_TOKEN, tokenizer, return_tensors="pt", image_token_number=img_token_num)
            elif timestamp[1] == 'video_audio':
                ids = tokenizer_image_audio_token(DEFAULT_VIDEO_AUDIO_TOKEN + DEFAULT_SEG_TOKEN, tokenizer, return_tensors="pt", image_token_number=img_token_num)
            else:
                ids = tokenizer_image_audio_token(user_conversations[num_text] + DEFAULT_SEG_TOKEN, tokenizer, return_tensors="pt", image_token_number=img_token_num)
                num_text += 1
            user_input_ids.append(ids)
            accumulated_token_num += len(ids)
            num_token.append(accumulated_token_num)
        
        max_token_num = accumulated_token_num
        preserved_num_token = copy.deepcopy(num_token)
        user_input_ids = torch.cat(user_input_ids)
        agent_input_ids = torch.full((len(user_input_ids)+512,), tokenizer.pad_token_id, device=user_input_ids.device, dtype=user_input_ids.dtype)
        targets = torch.full((len(user_input_ids)+512,), IGNORE_INDEX, device=user_input_ids.device, dtype=user_input_ids.dtype)
        informative_label = torch.full((len(user_input_ids)+512,), IGNORE_INDEX, device=user_input_ids.device, dtype=user_input_ids.dtype)
        total_len = len(user_input_ids)
        
        gpt_conversations = []
        audio_answer = []
        for i, sentence in enumerate(source):
            if sentence['from'] == 'gpt':
                gpt_conversations.append(sentence)
                if i == 0 or source[i-1]['from'] == 'gpt':
                    audio_answer.append(False)
                else:
                    audio_answer.append(True)
        
        # Proactive数据的answer一定跟在image segment后面，并且时间不早于question的时间，每条数据最多仅有一个question
        image_num_token = [num_token[i] for i, timestamp in enumerate(user_input_timestamp) if timestamp[1] == 'image']
        image_insert_timestamps = [timestamp[0] for i, timestamp in enumerate(user_input_timestamp) if timestamp[1] == 'image']
        image_num_token = [system_prompt_len] + image_num_token # 错开了一位,如果时间比当前image早，就跟在上一个image后面
        image_insert_timestamps = image_insert_timestamps + [1000000] # prevent overflow

        audio_num_token = [num_token[i] for i, timestamp in enumerate(user_input_timestamp) if timestamp[1] == 'video_audio']
        audio_insert_timestamps = [timestamp[0] for i, timestamp in enumerate(user_input_timestamp) if timestamp[1] == 'video_audio']
        audio_num_token = [system_prompt_len] + audio_num_token # 错开了一位,如果时间比当前image早，就跟在上一个image后面
        audio_insert_timestamps = audio_insert_timestamps + [1000000] # prevent overflow

        text_num_token = [num_token[i] for i, timestamp in enumerate(user_input_timestamp) if timestamp[1] == 'text']
        text_insert_timestamps = [timestamp[0] for i, timestamp in enumerate(user_input_timestamp) if timestamp[1] == 'text']
        text_num_token = [system_prompt_len] + text_num_token # 错开了一位,如果时间比当前image早，就跟在上一个image后面
        text_insert_timestamps = text_insert_timestamps + [1000000] # prevent overflow

        
        current_time = -1
        for sentence, audio_answer_flag in zip(gpt_conversations, audio_answer):
            user_time_id = 0
            if audio_answer_flag == False:
                insert_timestamps = image_insert_timestamps
                num_token = image_num_token
                start_token = "☟"
            else:
                if flag:
                    insert_timestamps = audio_insert_timestamps
                    num_token = audio_num_token
                    start_token = "☞"
                else:
                    insert_timestamps = text_insert_timestamps
                    num_token = text_num_token
                    start_token = "☜"
            while insert_timestamps[user_time_id] < current_time:
                user_time_id += 1
            while sentence['time'] >= insert_timestamps[user_time_id]:
                user_time_id += 1
            current_time = sentence['time']
            # if sentence['time'] - start_time < insert_timestamps[user_time_id]
            ids = tokenizer_image_audio_token(start_token + sentence['value'] + "<|im_end|>", tokenizer, return_tensors="pt")
            max_token_num = max(max_token_num, num_token[user_time_id] + len(ids))
            agent_input_ids[num_token[user_time_id]: num_token[user_time_id] + len(ids)] = ids
            targets[num_token[user_time_id]: num_token[user_time_id] + len(ids)] = ids

        # proactive data should apply supervision on informative head
        if not no_visual_proactive[j]:
            visual_seg = [preserved_num_token[i] - 2 for i, timestamp in enumerate(user_input_timestamp) if timestamp[1] == 'image']
            for token_id in visual_seg:
                informative_label[token_id] = 0
            for sentence, audio_answer_flag in zip(gpt_conversations, audio_answer):
                if not audio_answer_flag:
                    begin_time = (sentence['timespan'][0] + sentence['timespan'][1]) / 2
                    pos_token = [preserved_num_token[i] - 2 for i, timestamp in enumerate(user_input_timestamp) if timestamp[1] == 'image' and timestamp[0] >= begin_time and timestamp[0] <= sentence['timespan'][1]]
                    for token_id in pos_token:
                        informative_label[token_id] = 1

        agent_input_ids = agent_input_ids[:max_token_num]
        targets = targets[:max_token_num]
        informative_label = informative_label[:max_token_num]
        if max_token_num > len(user_input_ids):
            user_input_ids = torch.cat([user_input_ids, torch.full((max_token_num - len(user_input_ids),), tokenizer.pad_token_id, device=user_input_ids.device, dtype=user_input_ids.dtype)])
        
        if flag:
            audio_seg = [preserved_num_token[i] for i, timestamp in enumerate(user_input_timestamp) if timestamp[1] == 'video_audio']
            for token_id in audio_seg:
                if targets[token_id] == IGNORE_INDEX:
                    targets[token_id] = tokenizer.pad_token_id
        assert len(user_input_ids) == len(agent_input_ids)
        all_user_input_ids.append(user_input_ids)
        all_agent_input_ids.append(agent_input_ids)
        all_targets.append(targets)
        all_informative_labels.append(informative_label)

    all_user_input_ids = torch.stack(all_user_input_ids)
    all_agent_input_ids = torch.stack(all_agent_input_ids)
    all_targets = torch.stack(all_targets)
    informative_labels = torch.stack(all_informative_labels)

    return dict(
        all_user_input_ids=all_user_input_ids,
        all_agent_input_ids=all_agent_input_ids,
        labels=all_targets,
        informative_labels=informative_labels,
    )



def _get_rawvideo_dec(
    video_path,
    image_processor,
    max_frames=32,
    min_frames=4,
    video_framerate=3,
    audio_segment_len=1.0,
    max_video_audio_segment=6,
    start_time=None,
    end_time=None,
    image_aspect_ratio="pad",
):
    if os.path.exists(video_path):
        vreader = VideoReader(video_path, ctx=cpu(0))
    else:
        print(video_path)
        raise FileNotFoundError

    fps = vreader.get_avg_fps()
    f_start = 0 if start_time is None else int(start_time * fps)
    f_end = int(min(1000000000 if end_time is None else end_time * fps, len(vreader) - 1))
    num_frames = f_end - f_start + 1
    if num_frames > 0:
        # T x 3 x H x W
        sample_fps = int(video_framerate)
        t_stride = int(round(float(fps) / sample_fps))

        all_pos = list(range(f_start, f_end + 1, t_stride))
        if len(all_pos) > max_frames:
            sample_pos = [
                all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=max_frames, dtype=int)
            ]
        elif len(all_pos) < min_frames:
            sample_pos = np.linspace(f_start, f_end, num=min_frames, dtype=int)
            sample_pos = sample_pos.tolist()
        else:
            sample_pos = all_pos

        patch_images = [Image.fromarray(f) for f in vreader.get_batch(sample_pos).asnumpy()]
        sample_time = [frame_id / fps for frame_id in sample_pos]

        if image_aspect_ratio == "pad":

            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            patch_images = [
                expand2square(i, tuple(int(x * 255) for x in image_processor.image_mean))
                for i in patch_images
            ]
            patch_images = [
                image_processor.preprocess(i, return_tensors="pt")["pixel_values"][0]
                for i in patch_images
            ]
        else:
            patch_images = [
                image_processor.preprocess(i, return_tensors="pt")["pixel_values"][0]
                for i in patch_images
            ]

        slice_len = len(patch_images)
        del vreader

        try:
            audio_reader = AudioReader(video_path, ctx=cpu(0), sample_rate=16000)
            audio_tensor = torch.from_numpy(audio_reader._array)
            del audio_reader
        except:
            audio_tensor = None

        if audio_tensor is None:
            return patch_images, slice_len, None, [0 for _ in range(slice_len)], sample_time
        else:
            # 自适应调整audio的长度, 但是仍然保持每段audio的长度是固定的，这样有较好的test time适应性
            num_audio_seg = round(((sample_pos[1] - sample_pos[0]) / fps) / audio_segment_len) # 防止0.99
            num_audio_seg = int(max(min(num_audio_seg, max_video_audio_segment), 1)) # [1, max_video_audio_segment]

            # 裁剪为audio片段
            # [frame][audio][audio][frame][audio][auido]
            audio_patch = []
            all_num_audio_seg = []
            for idx in sample_pos:
                start = int(max(idx / fps * 16000, 0))
                break_flag = 0
                for j in range(num_audio_seg):
                    end = int(start + 16000 * audio_segment_len)
                    if end > audio_tensor.shape[1]:
                        break_flag = 1
                        break
                    else:
                        audio_patch.append(audio_tensor[:, start: end])
                        start = end
                all_num_audio_seg.append(j if break_flag else j + 1)

            return patch_images, slice_len, audio_patch, all_num_audio_seg, sample_time
    else:
        print("video path: {} error.".format(video_path))
        raise FileNotFoundError



class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()

        dataset_list = DataConfig[str(data_args.dataset_use)]
        print(dataset_list)

        self.max_length = MAX_IMAGE_LENGTH
        list_data_dict = []
        for i in dataset_list:
            # list_data_dict += json.load(open(i["chat_path"], "r"))
            data_ratio = i.get("data_ratio", DEFAULT_DATA_RATIO)
            data_i = json.load(open(i["chat_path"], "r"))
            len_data_i = len(data_i)
            data_i = data_i[:int(len_data_i * data_ratio)]
            streaming = i.get("streaming", False)
            if streaming:
                data_type = 'streaming'
            else:
                data_type = 'offline'
            if 'interrupt' in i["chat_path"]:
                data_type = 'interrupt'
            if i.get("gesture", False):
                data_type = 'gesture'
            if i.get("audio_proactive", False):
                data_type = 'audio_proactive'
            no_visual_proactive = i.get('no_visual_proactive', False)
            audio_folder_path = i.get("audio_folder_path", None)
            image_folder_path = i.get("image_folder_path", None)
            video_folder_path = i.get("video_folder_path", None)
            for da in data_i:
                da['data_type'] = data_type
                da['no_visual_proactive'] = no_visual_proactive
                if audio_folder_path is not None:
                    if 'audio' in da.keys():
                        if type(da['audio']) is str:
                            da['audio'] = [da['audio']]
                        da['audio'] = [os.path.join(audio_folder_path, file) for file in da['audio']]
                if image_folder_path is not None:
                    if 'image' in da.keys():
                        if type(da['image']) is str:
                            da['image'] = [da['image']]
                        da['image'] = [os.path.join(image_folder_path, file) for file in da['image']]
                if video_folder_path is not None:
                    if 'video' in da.keys(): # one sample should only have one video
                        da['video'] = os.path.join(video_folder_path, da['video'])

            list_data_dict += data_i

        random.shuffle(list_data_dict)

        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.use_informative_head = data_args.use_informative_head
        print("max video frame num: %d" % (MAX_IMAGE_LENGTH * self.data_args.pooling_size * self.data_args.pooling_size))

    def __len__(self):
        return len(self.list_data_dict)

    # @property
    # def lengths(self):
    #     length_list = []
    #     for sample in self.list_data_dict:
    #         img_tokens = 128 if 'image' in sample else 0
    #         length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
    #     return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            cur_len = cur_len if ("image" in sample or "video" in sample) else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        try:
            sources = self.list_data_dict[i]
            data_type = sources.get('data_type', 'offline') # ['offline', 'streaming', 'interrupt']
            if isinstance(i, int):
                sources = [sources]
            assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
            image, audio, video, video_audio = None, None, None, None
            image_token_num = 1
            patch_num = [1]
            audio_segment_num = [1]
            if "image" in sources[0] and data_type != 'interrupt':
                image_file = self.list_data_dict[i]["image"]
                if type(image_file) is str:
                    image_file = [image_file]
                processor = self.data_args.image_processor
                if "height" in processor.size.keys():
                    image_size = processor.size["height"]
                elif "shortest_edge" in processor.size.keys():
                    image_size = processor.size["shortest_edge"]
                else:
                    raise NotImplementedError(f"Please use correct key to use processor size!")

                image = [Image.open(file).convert("RGB") for k, file in enumerate(image_file)]
                if self.data_args.image_aspect_ratio == "pad":

                    def expand2square(pil_img, background_color):
                        width, height = pil_img.size
                        if width == height:
                            return pil_img
                        elif width > height:
                            result = Image.new(pil_img.mode, (width, width), background_color)
                            result.paste(pil_img, (0, (width - height) // 2))
                            return result
                        else:
                            result = Image.new(pil_img.mode, (height, height), background_color)
                            result.paste(pil_img, ((height - width) // 2, 0))
                            return result

                    image = [
                        expand2square(i, tuple(int(x * 255) for x in processor.image_mean))
                        for i in image
                    ]
                image_patches, patch_num = [], []
                for k, img in enumerate(image):
                    img, p_num = dynamic_preprocess(
                        img,
                        min_num=self.data_args.min_dynamic_patch,
                        max_num=self.data_args.max_dynamic_patch,
                        image_size=image_size,
                        use_thumbnail=self.data_args.use_thumbnail,
                    )
                    image_patches += img
                    patch_num += p_num
                assert len(image_patches) == sum(patch_num)
                image = image_patches
                image = [
                    processor.preprocess(i, return_tensors="pt")["pixel_values"][0]
                    for i in image
                ]
            
            if "image" in sources[0] and data_type == 'interrupt':
                image_file = self.list_data_dict[i]["image"]
                if type(image_file) is list:
                    image_file = image_file[0]
                processor = self.data_args.image_processor
                image = Image.open(image_file).convert("RGB")
                image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

            audio_flag = False
            if "audio" in sources[0]:
                audio_flag = True
                audio_file = self.list_data_dict[i]["audio"]
                if type(audio_file) is str:
                    audio_file = [audio_file]

                assert len(audio_file) > 0, "audio_file为列表时不能为空"
                audio = []
                audio_for_llm_lens = []
                audio_length = []
                audio_segment_num = []
                for file in audio_file:
                    waveform, sample_rate = torchaudio.load(file)
                    if sample_rate != 16000:
                        waveform = torchaudio.transforms.Resample(
                            orig_freq=sample_rate, new_freq=16000
                        )(waveform)
                    if waveform.shape[1] % int(16000 * self.data_args.audio_segment_len) != 0:
                        segment_length = int(16000 * self.data_args.audio_segment_len)
                        padding_length = segment_length - (waveform.shape[1] % segment_length)
                        waveform = F.pad(waveform, (0, padding_length), "constant", 0)
                    segment_num = waveform.shape[1] // int(16000 * self.data_args.audio_segment_len)
                    if segment_num == 0 or segment_num > 32:
                        print(file, 'too long or too short')
                        assert False
                    audio_segment_num.append(segment_num)
                    for seg_id in range(segment_num):
                        a, a_llm = self.data_args.audio_processor.process(waveform[:, seg_id * int(16000 * self.data_args.audio_segment_len): (seg_id + 1) * int(16000 * self.data_args.audio_segment_len)])
                        audio.append(a)
                        audio_for_llm_lens.append(a_llm)
                        audio_length.append(a.shape[0])
            video_flag = False
            video_audio_segment_num = 0
            video_start_time = None
            video_end_time = None
            if "video" in sources[0]:
                video_flag = True
                video_file = self.list_data_dict[i]["video"]
                processor = self.data_args.image_processor
                if "height" in processor.size.keys():
                    image_size = processor.size["height"]
                elif "shortest_edge" in processor.size.keys():
                    image_size = processor.size["shortest_edge"]
                else:
                    raise NotImplementedError(f"Please use correct key to use processor size!")
                video_start_time = sources[0].get('video_start_time', None)
                video_end_time = sources[0].get('video_end_time', None)
                if video_start_time is not None:
                    video_start_time = int(video_start_time)
                    video_start_time = video_start_time if video_start_time >= 0 else 0
                else:
                    video_start_time = 0

                if video_end_time is not None:
                    video_end_time = int(video_end_time) + 1
                    video_end_time = video_end_time if video_end_time >= 0 else 0

                image, image_token_num, video_audio, video_audio_segment_num, sample_time = _get_rawvideo_dec(
                    video_file,
                    self.data_args.image_processor,
                    max_frames=MAX_IMAGE_LENGTH * self.data_args.pooling_size * self.data_args.pooling_size,
                    min_frames=MIN_IMAGE_LENGTH * self.data_args.minimal_image_ratio,
                    image_aspect_ratio=self.data_args.image_aspect_ratio,
                    audio_segment_len=self.data_args.audio_segment_len,
                    video_framerate=self.data_args.video_framerate,
                    max_video_audio_segment=self.data_args.max_video_audio_segment,
                    start_time=video_start_time,
                    end_time=video_end_time
                )
                if video_audio is not None and len(video_audio) > 0:
                    processed_video_audio, video_audio_for_llm_lens, video_audio_length = [], [], []
                    for file in video_audio:
                        a, a_llm = self.data_args.audio_processor.process(file)
                        processed_video_audio.append(a)
                        video_audio_for_llm_lens.append(a_llm)
                        video_audio_length.append(a.shape[0])


            if data_type == 'offline':
                sources = preprocess_multimodal_offline(
                    copy.deepcopy([e["conversations"] for e in sources]),
                    self.data_args,
                    patch_num=patch_num,
                    audio_segment_num=audio_segment_num,
                    video_audio_segment_num=video_audio_segment_num,
                )

                data_dict = preprocess_qwen2p5_moshi_offline_data(
                    sources,
                    self.tokenizer,
                    img_token_num=DEFAULT_IMAGE_TOKEN_NUMBER // self.data_args.pooling_size // self.data_args.pooling_size,
                )
                
            elif data_type == 'streaming':
                data_dict = preprocess_qwen2p5_moshi_online_data(
                    copy.deepcopy([e["conversations"] for e in sources]),
                    self.tokenizer,
                    img_token_num=DEFAULT_IMAGE_TOKEN_NUMBER // self.data_args.pooling_size // self.data_args.pooling_size,
                    sample_time=sample_time,
                    proactive_flags=[e["proactive"] for e in sources],
                    video_audio_segment_num=video_audio_segment_num,
                )

            elif data_type == 'audio_proactive':
                data_dict = preprocess_qwen2p5_moshi_audio_proactive_data(
                    copy.deepcopy([e["conversations"] for e in sources]),
                    self.tokenizer,
                    sample_time=sample_time,
                    img_token_num=DEFAULT_IMAGE_TOKEN_NUMBER // self.data_args.pooling_size // self.data_args.pooling_size,
                    use_audios=[e.get('use_audio', False) for e in sources],
                    video_audio_segment_num=video_audio_segment_num,
                    no_visual_proactive=[e["no_visual_proactive"] for e in sources]
                )
                if not sources[0].get('use_audio', False):
                    video_audio = None

            

            if audio_flag:
                data_dict["audio_lengths"] = audio_length
                data_dict["audio_lengths_for_llm"] = audio_for_llm_lens

            if video_flag and video_audio is not None and len(video_audio) > 0:
                data_dict["video_audio_lengths"] = video_audio_length
                data_dict["video_audio_lengths_for_llm"] = video_audio_for_llm_lens

            if isinstance(i, int):
                data_dict['all_user_input_ids'] = data_dict['all_user_input_ids'][0]
                data_dict['all_agent_input_ids'] = data_dict['all_agent_input_ids'][0]
                if torch.any(data_dict['all_agent_input_ids'] < 0):
                    assert False, "There is a specitial token in response!"
                if 'labels' in data_dict:
                    data_dict['labels'] = data_dict['labels'][0]
                if 'informative_labels' in data_dict:
                    data_dict['informative_labels'] = data_dict['informative_labels'][0]
                
            # image exist in the data
            if "image" in self.list_data_dict[i] or "video" in self.list_data_dict[i]:
                data_dict["image"] = image
            elif self.data_args.is_multimodal:
                # image does not exist in the data, but the model is multimodal
                crop_size = self.data_args.image_processor.crop_size
                data_dict["image"] = [torch.zeros(3, crop_size["height"], crop_size["width"])]
            if "audio" in self.list_data_dict[i]:
                data_dict["audio"] = audio
            elif self.data_args.is_multimodal:
                data_dict["audio"] = [torch.zeros(98, 80)]
                data_dict["audio_lengths"] = [98]
                data_dict["audio_lengths_for_llm"] = [12]
            if "video" in self.list_data_dict[i] and video_audio is not None and len(video_audio) > 0:
                data_dict['video_audio'] = processed_video_audio
                del video_audio
            elif self.data_args.is_multimodal:
                data_dict["video_audio"] = [torch.zeros(98, 80)]
                data_dict["video_audio_lengths"] = [98]
                data_dict["video_audio_lengths_for_llm"] = [12]
            
            assert len(data_dict['all_user_input_ids']) <= self.tokenizer.model_max_length, "Input too long!"
            single_image_tokens = DEFAULT_IMAGE_TOKEN_NUMBER // self.data_args.pooling_size // self.data_args.pooling_size
            assert max((data_dict['all_user_input_ids'] == IMAGE_TOKEN_INDEX).sum() // single_image_tokens, 1) == len(data_dict["image"]), "Invalid Image Number: %d vs %d" % (int(max((data_dict['all_user_input_ids'] == IMAGE_TOKEN_INDEX).sum() // single_image_tokens, 1)), len(data_dict["image"]))
            assert max((data_dict['all_user_input_ids'] == AUDIO_TOKEN_INDEX).sum() // DEFAULT_AUDIO_TOKEN_NUMBER, 1) == len(data_dict["audio"]), "Invalid Audio Number: %d vs %d" % (int(max((data_dict['all_user_input_ids'] == AUDIO_TOKEN_INDEX).sum() // DEFAULT_AUDIO_TOKEN_NUMBER, 1)), len(data_dict["audio"]))
            assert max((data_dict['all_user_input_ids'] == VIDEO_AUDIO_TOKEN_INDEX).sum() // DEFAULT_AUDIO_TOKEN_NUMBER, 1) == len(data_dict["video_audio"]), "Invalid Video_Audio Number: %d vs %d" % (int(max((data_dict['all_user_input_ids'] == VIDEO_AUDIO_TOKEN_INDEX).sum() // DEFAULT_AUDIO_TOKEN_NUMBER, 1)), len(data_dict["video_audio"]))


        except Exception as e:
            print(f'Error in dataset: {e}')
            print(self.list_data_dict[i])
            print('Using a random data instead.')
            data_dict = self.__getitem__(random.choice(list(range(len(self)))))
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:

        all_user_input_ids = [instance["all_user_input_ids"] for instance in instances]
        all_agent_input_ids = [instance["all_agent_input_ids"] for instance in instances]
        if 'labels' in instances[0]:
            labels = [instance["labels"] for instance in instances]
        else:
            labels = None
        if 'informative_labels' in instances[0]:
            informative_labels = [instance["informative_labels"] for instance in instances]
        else:
            informative_labels = None


        input_lens = [len(input_ids) for input_ids in all_user_input_ids]
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in all_user_input_ids:
                input_id[input_id == self.tokenizer.eos_token_id] = -300
            for input_id in all_agent_input_ids :
                input_id[input_id == self.tokenizer.eos_token_id] = -300

        all_user_input_ids = torch.nn.utils.rnn.pad_sequence(
            all_user_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        all_agent_input_ids = torch.nn.utils.rnn.pad_sequence(
            all_agent_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        if labels is not None:
            labels = torch.nn.utils.rnn.pad_sequence(
                labels, batch_first=True, padding_value=IGNORE_INDEX
            )
        if informative_labels is not None:
            informative_labels = torch.nn.utils.rnn.pad_sequence(
                informative_labels, batch_first=True, padding_value=IGNORE_INDEX
            )

        all_user_input_ids = all_user_input_ids[:, : self.tokenizer.model_max_length]
        all_agent_input_ids = all_agent_input_ids[:, : self.tokenizer.model_max_length]
        attention_mask = torch.ones_like(all_agent_input_ids, dtype=torch.bool)
        for i, input_len in enumerate(input_lens):
            attention_mask[i, input_len:] = False
        if labels is not None:
            labels = labels[:, : self.tokenizer.model_max_length]
        if informative_labels is not None:
            informative_labels = informative_labels[:, : self.tokenizer.model_max_length]

        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in all_user_input_ids:
                input_id[input_id == -300] = self.tokenizer.eos_token_id
            for input_id in all_agent_input_ids:
                input_id[input_id == -300] = self.tokenizer.eos_token_id

        batch = dict(
            user_input_ids=all_user_input_ids,
            input_ids=all_agent_input_ids,
            labels=labels,
            informative_labels=informative_labels,
            attention_mask=attention_mask,
        )

        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]

            new_images = []
            for image in images:
                if type(image) is list:
                    for i in image:
                        new_images.append(i)
                else:
                    new_images.append(image)
            images = new_images

            if all(x is not None and x.shape == images[0].shape for x in images):
                batch["images"] = torch.stack(images)
            else:
                batch["images"] = images

        batch["audios"] = {}
        if "audio" in instances[0]:
            audios = [instance["audio"] for instance in instances]
            audio_lengths = [instance["audio_lengths"] for instance in instances]
            audio_lengths_for_llm = [instance["audio_lengths_for_llm"] for instance in instances]

            new_audios = []
            new_audio_lengths = []
            new_audio_lengths_for_llm = []
            for i, audio in enumerate(audios):
                length = audio_lengths[i]
                length_for_llm = audio_lengths_for_llm[i]
                if type(audio) is list:
                    for j, a in enumerate(audio):
                        new_audios.append(a)
                        new_audio_lengths.append(length[j])
                        new_audio_lengths_for_llm.append(length_for_llm[j])
                else:
                    new_audios.append(audio)
                    new_audio_lengths.append(length)
                    new_audio_lengths_for_llm.append(length_for_llm)
            audios = new_audios
            audios = pad_sequence(audios, batch_first=True, padding_value=0)

            batch["audios"]["audios"] = audios
            batch["audios"]["lengths"] = torch.tensor(new_audio_lengths)
            batch["audios"]["lengths_for_llm"] = torch.tensor(new_audio_lengths_for_llm)

        batch["video_audios"] = {}
        if "video_audio" in instances[0]:
            audios = [instance["video_audio"] for instance in instances]
            audio_lengths = [instance["video_audio_lengths"] for instance in instances]
            audio_lengths_for_llm = [instance["video_audio_lengths_for_llm"] for instance in instances]

            new_audios = []
            new_audio_lengths = []
            new_audio_lengths_for_llm = []
            for i, audio in enumerate(audios):
                length = audio_lengths[i]
                length_for_llm = audio_lengths_for_llm[i]
                if type(audio) is list:
                    for j, a in enumerate(audio):
                        new_audios.append(a)
                        new_audio_lengths.append(length[j])
                        new_audio_lengths_for_llm.append(length_for_llm[j])
                else:
                    new_audios.append(audio)
                    new_audio_lengths.append(length)
                    new_audio_lengths_for_llm.append(length_for_llm)
            audios = new_audios
            audios = pad_sequence(audios, batch_first=True, padding_value=0)

            batch["video_audios"]["audios"] = audios
            batch["video_audios"]["lengths"] = torch.tensor(new_audio_lengths)
            batch["video_audios"]["lengths_for_llm"] = torch.tensor(new_audio_lengths_for_llm)
        batch["pad_token_id"] = self.tokenizer.pad_token_id

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    # print(f'width: {width}, height: {height}, best_ratio: {best_ratio}')
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images, [len(processed_images)]

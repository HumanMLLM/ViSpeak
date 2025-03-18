
import argparse
import itertools
import json
import os
import random
import time
from functools import partial
import re
from evaluate_tokenizer import EvaluationTokenizer
import editdistance as ed
import torch
from transformers.pipelines.audio_utils import ffmpeg_read
import requests
from whisper_normalizer.english import EnglishTextNormalizer
from whisper_normalizer.basic import BasicTextNormalizer
from cn_tn import TextNorm
import zhconv
import torchaudio
import torch.nn.functional as F

from vispeak.constants import (
    DEFAULT_AUDIO_TOKEN,
    DEFAULT_SEG_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_AUDIO_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    MAX_IMAGE_LENGTH,
)
from vispeak.conversation import SeparatorStyle, conv_templates
from vispeak.model.builder import load_pretrained_model
from vispeak.util.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    tokenizer_image_audio_token,
    tokenizer_image_token,
)
from vispeak.util.data_utils import SYSTEM_PROMTP, preprocess_qwen2p5_moshi_offline_data

english_normalizer = EnglishTextNormalizer()
chinese_normalizer = TextNorm(
        to_banjiao = False,
        to_upper = False,
        to_lower = False,
        remove_fillers = False,
        remove_erhua =False,
        check_chars = False,
        remove_space = False,
        cc_mode = '',
    )
basic_normalizer = BasicTextNormalizer()

from tqdm import tqdm


PUNCS = '!,.?;:'

ds_collections = {
    'librispeech': {'path': '/mnt/data/shenghao/datasets/librispeech/librispeech_eval.jsonl','language': 'en', "data_root": "/mnt/workspace/shenghao/datasets"},
    # 'aishell2': {'path': 'asr/aishell2_eval.jsonl', 'language': 'zh'},
    'cv15_en': {'path': '/mnt/data/qize.yqz/datasets/audio/eval/cv15_asr_en_eval.jsonl', 'language': 'en', "data_root": "/mnt/data/qize.yqz/datasets/audio/common_voice_15_0"},
    'cv15_zh': {'path': '/mnt/data/qize.yqz/datasets/audio/eval//cv15_asr_zh_eval.jsonl', 'language': 'zh', "data_root": "/mnt/data/qize.yqz/datasets/audio/common_voice_15_0"},
    # 'cv15_yue': {'path': 'asr/cv15_asr_yue_eval.jsonl', 'language': 'yue'},
    # 'cv15_fr': {'path': 'asr/cv15_asr_fr_eval.jsonl', 'language': 'fr'},
    'fluers_zh': {'path': '/mnt/data/qize.yqz/datasets/audio/eval/fleurs_asr_zh_eval.jsonl', 'language': 'zh', "data_root": "/mnt/data/qize.yqz/datasets/audio"},
    'fluers_en': {'path': '/mnt/data/qize.yqz/datasets/audio/eval/fleurs_asr_en_eval.jsonl', 'language': 'en', "data_root": "/mnt/data/qize.yqz/datasets/audio"},
    'wenet_net': {'path': '/mnt/data/qize.yqz/datasets/audio/wenetspeech/test/TEST_NET.jsonl', 'language': 'zh', "data_root": "/mnt/data/qize.yqz/datasets/audio/wenetspeech/test"},
    'wenet_meet': {'path': '/mnt/data/qize.yqz/datasets/audio/wenetspeech/test/TEST_MEETING.jsonl', 'language': 'zh', "data_root": "/mnt/data/qize.yqz/datasets/audio/wenetspeech/test"},

}


class AudioDataset(torch.utils.data.Dataset):

    def __init__(self, ds):
        path = ds['path']
        self.data_root = ds['data_root']
        self.datas = open(path).readlines()
        self.lang = ds['language']

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = json.loads(self.datas[idx].strip())
        audio = os.path.join(self.data_root, data['audio'])
        source = data['source'] if 'source' in data else "wenet"
        # prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>"+data['prompt']
        if self.lang =="en":
            prompt = "Give me the transcription of the speech you heard"
        else:
             prompt = "请将语音内容转换成文字，并以文字形式呈现出来。"
        gt = data['gt'] if 'gt' in data else data['text']

        return {
            'audio': audio,
            'prompt': prompt,
            'source': source,
            'gt': gt
        }

def read_audio(audio_path):
    if audio_path.startswith("http://") or audio_path.startswith("https://"):
        # We need to actually check for a real protocol, otherwise it's impossible to use a local file
        # like http_huggingface_co.png
        inputs = requests.get(audio_path).content
    else:
        with open(audio_path, "rb") as f:
            inputs = f.read()
    return inputs

# def collate_fn(inputs, processor):
#     input_texts = [_['prompt'] for _ in inputs]
#     source = [_['source'] for _ in inputs]
#     gt = [_['gt'] for _ in inputs]
#     audio_path = [_['audio'] for _ in inputs]
#     input_audios = [ffmpeg_read(read_audio(_['audio']),sampling_rate=processor.feature_extractor.sampling_rate) for _ in inputs]
#     inputs = processor(text=input_texts, audios=input_audios, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)
#     return inputs, audio_path, source, gt


def collate_fn(inputs):
    input_texts = [_['prompt'] for _ in inputs]
    source = [_['source'] for _ in inputs]
    gt = [_['gt'] for _ in inputs]
    audio_path = [_['audio'] for _ in inputs]
    # input_audios = [ffmpeg_read(read_audio(_['audio']),sampling_rate=processor.feature_extractor.sampling_rate) for _ in inputs]
    # inputs = processor(text=input_texts, audios=input_audios, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)
    return input_texts, audio_path, source, gt

class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size,
                                                      self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)

def remove_sp(text, language):
    gt = re.sub(r"<\|.*?\|>", " ", text)
    gt = re.sub(rf"\s+", r" ", gt)  # 将文本中的连续空格替换为单个空格
    gt = re.sub(f" ?([{PUNCS}])", r"\1", gt)
    gt = gt.lstrip(" ")
    if language == "zh":
        gt = re.sub(rf"\s+", r"", gt)
    return gt

def compute_wer(refs, hyps, language):
    distance = 0
    ref_length = 0
    tokenizer = EvaluationTokenizer(
            tokenizer_type="none",
            lowercase=True,
            punctuation_removal=True,
            character_tokenization=False,
        )
    for i in range(len(refs)):
        ref = refs[i]
        pred = hyps[i]
        if language in ["yue"]:
            ref = zhconv.convert(ref, 'zh-cn')
            pred = zhconv.convert(pred, 'zh-cn')
        if language in ["en"]:
            ref = english_normalizer(ref)
            pred = english_normalizer(pred)
        if language in ["zh"]:
            ref = chinese_normalizer(ref)
            pred = chinese_normalizer(pred)
        else:
            ref = basic_normalizer(ref)
            pred = basic_normalizer(pred)
        ref_items = tokenizer.tokenize(ref).split()
        pred_items = tokenizer.tokenize(pred).split()
        if language in ["zh", "yue"]:
            ref_items = [x for x in "".join(ref_items)]
            pred_items = [x for x in "".join(pred_items)]
        if i==0:
            print(f"ref: {ref}")
            print(f"pred: {pred}")
            print(f"ref_items:\n{ref_items}\n{len(ref_items)}\n{ref_items[0]}")
            print(f"pred_items:\n{pred_items}\n{len(ref_items)}\n{ref_items[0]}")
        distance += ed.eval(ref_items, pred_items)
        ref_length += len(ref_items)
    return distance/ref_length


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument('--dataset', type=str, default='librispeech')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument("--model_type", type=str, default="qwen2p5_instruct")
    parser.add_argument("--conv_mode", type=str, default="qwen2p5_instruct")
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    model_path = args.model_path
    model_base = args.model_base
    conv_mode = args.conv_mode

    # Sampling Parameter
    temperature = 0.01
    top_p = None
    num_beams = 1
    # disable_torch_init()

    model_path = os.path.expanduser(model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base, model_name, args.model_type
    )

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    image_processor = vision_tower.image_processor

    audio_encoder = model.get_audio_encoder()
    audio_encoder.to(dtype=torch.float16)
    audio_processor = audio_encoder.audio_processor
    model.eval()

    # model = Qwen2AudioForConditionalGeneration.from_pretrained(
    #     args.checkpoint, device_map='cuda', torch_dtype='auto', trust_remote_code=True).eval()

    # processor = AutoProcessor.from_pretrained(args.checkpoint)
    # processor.tokenizer.padding_side = 'left'

    random.seed(args.seed)
    dataset = AudioDataset(
        ds=ds_collections[args.dataset],
    )
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=InferenceSampler(len(dataset)),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=partial(collate_fn),
    )

    gts = []
    sources = []
    rets = []
    audio_paths = []
    for _, (inputs, audio_path, source, gt) in tqdm(enumerate(data_loader), total=len(dataset)):
        waveform, sample_rate = torchaudio.load(audio_path[0])
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=16000
            )(waveform)
        if waveform.shape[1] % 16000 != 0:
            segment_length = 16000
            padding_length = segment_length - (waveform.shape[1] % segment_length)
            waveform = F.pad(waveform, (0, padding_length), "constant", 0)
        segment_num = waveform.shape[1] // 16000
        if segment_num == 0 or segment_num > 32:
            print(audio_path[0], 'too long or too short')
            assert False
        audio = []
        audio_for_llm_lens = []
        audio_length = []
        for seg_id in range(segment_num):
            a, a_llm = audio_processor.process(waveform[:, seg_id * 16000: (seg_id + 1) * 16000])
            audio.append(a)
            audio_for_llm_lens.append(a_llm)
            audio_length.append(a.shape[0])

        audio = torch.stack(audio, dim=0)
        audio_length = torch.tensor(audio_length)
        audio_for_llm_lens = torch.tensor(audio_for_llm_lens)
        audios = dict()
        audios["audios"] = audio.half().cuda()
        audios["lengths"] = audio_length.half().cuda()
        audios["lengths_for_llm"] = audio_for_llm_lens.cuda()
        # inputs['input_ids'] = inputs['input_ids'].to('cuda')

        image_tensor = torch.zeros((1, 3, 448, 448)).to(dtype=model.dtype, device="cuda")

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

        system_prompt = SYSTEM_PROMTP['audio']
        print(inputs[0])

        user_input_timestamp = []
        for i in range(len(audio)):
            user_input_timestamp.append([i, 'audio'])
        user_input_timestamp.append([i+1, 'text'])

        # sort with timestamp
        user_input_timestamp.sort(key=lambda x: x[0])
        user_input_ids = []
        num_token = []
        
        # system prompt
        ids = tokenizer_image_audio_token(system_prompt + DEFAULT_SEG_TOKEN, tokenizer, return_tensors="pt")
        user_input_ids.append(ids)
        accumulated_token_num = len(ids)

        for timestamp in user_input_timestamp:
            if timestamp[1] == 'image':
                ids = tokenizer_image_audio_token(DEFAULT_IMAGE_TOKEN + DEFAULT_SEG_TOKEN, tokenizer, return_tensors="pt")
            elif timestamp[1] == 'audio':
                ids = tokenizer_image_audio_token(DEFAULT_AUDIO_TOKEN + DEFAULT_SEG_TOKEN, tokenizer, return_tensors="pt")
            elif timestamp[1] == 'video_audio':
                ids = tokenizer_image_audio_token(DEFAULT_VIDEO_AUDIO_TOKEN + DEFAULT_SEG_TOKEN, tokenizer, return_tensors="pt")
            else:
                ids = tokenizer_image_audio_token(inputs[0] + DEFAULT_SEG_TOKEN, tokenizer, return_tensors="pt")
            user_input_ids.append(ids)
            accumulated_token_num += len(ids)
            num_token.append(accumulated_token_num)
            
        user_input_ids = torch.cat(user_input_ids)
        start_inference_seg = [(i, num_token[i]) for i, timestamp in enumerate(user_input_timestamp) if timestamp[1] == 'text'][0]
        user_input_ids = user_input_ids.unsqueeze(0).cuda()

        keywords = ['<|im_end|>']
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, user_input_ids)

        sf_masks = torch.tensor([0]*len(image_tensor)).cuda()
        start_time = time.time()
        with torch.inference_mode():
            cont, responce_time = model.streaming_generate(
                user_input_ids,
                start_inference_seg=start_inference_seg,
                timestamps=[t[0] for t in user_input_timestamp],
                seg_token_id=tokenizer.convert_tokens_to_ids(DEFAULT_SEG_TOKEN),
                images=image_tensor,
                video_audios=video_audios,
                audios=audios,
                pad_token_id=tokenizer.pad_token_id,
                temperature=0.01,
                max_new_tokens=2048,
                padding_size=256,
                stopping_criteria=stopping_criteria,
            )
            outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
            if '☞' in outputs or '☜' in outputs or '☟' in outputs:
                outputs = outputs[1:]
        outputs = outputs.strip()


        print(audio_path[0],outputs,gt[0])
        
        gts.extend(gt)
        rets.extend([outputs])
        sources.extend(source)
        audio_paths.extend(audio_path)

    torch.distributed.barrier()

    world_size = torch.distributed.get_world_size()
    merged_gts = [None for _ in range(world_size)]
    merged_sources = [None for _ in range(world_size)]
    merged_responses = [None for _ in range(world_size)]
    merged_audio_paths = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(merged_gts, gts)
    torch.distributed.all_gather_object(merged_sources, sources)
    torch.distributed.all_gather_object(merged_responses, rets)
    torch.distributed.all_gather_object(merged_audio_paths, audio_paths)

    merged_gts = [_ for _ in itertools.chain.from_iterable(merged_gts)]
    merged_sources = [_ for _ in itertools.chain.from_iterable(merged_sources)]
    merged_audio_paths = [_ for _ in itertools.chain.from_iterable(merged_audio_paths)]
    merged_responses = [
        _ for _ in itertools.chain.from_iterable(merged_responses)
    ]

    if torch.distributed.get_rank() == 0:
        print(f"Evaluating {args.dataset} ...")

        results = []
        for gt, response, source, audio_path in zip(merged_gts, merged_responses, merged_sources, merged_audio_paths):
            results.append({
                'gt': gt,
                'response': response,
                'source': source,
                'audio_path': audio_path,
            })
        time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
        results_file = f'{args.dataset}_{time_prefix}.json'
        json.dump(results, open(results_file, 'w'), indent=2)
        results_dict = {}
        for item in tqdm(results):
            source = item["source"]
            results_dict.setdefault(source, []).append(item)
        lan = ds_collections[args.dataset]['language']
        for source in results_dict:
            refs, hyps = [], []
            results_list = results_dict[source]
            for result in results_list:
                gt = result["gt"]
                response = result["response"]
                gt = remove_sp(gt, lan)
                response = remove_sp(response, lan)
                refs.append(gt)
                hyps.append(response)
            wer = compute_wer(refs, hyps, lan)
            print(f"source: {source}  cnt: {len(refs)} wer: {wer:.4f}")


    torch.distributed.barrier()

"""
export PYTHONPATH=./

python -m torch.distributed.launch --use_env  --nproc_per_node 1 --nnodes 1 audio_eval/eval_asr.py \
   --model_path /mnt/data/qize.yqz/pretrained_models/VITA-MLLM/VITA-1.5/ 

"""

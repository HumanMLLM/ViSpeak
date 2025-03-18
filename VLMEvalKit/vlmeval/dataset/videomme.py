from huggingface_hub import snapshot_download
from ..smp import *
from .video_base import VideoBaseDataset
FAIL_MSG = 'Failed to obtain answer via API.'
import torch
import torch.nn.functional as F
import torchaudio

def unwrap_hf_pkl(pth, suffix='.mp4'):
    base_dir = os.path.join(pth, 'video_pkl/')
    target_dir = os.path.join(pth, 'video/')
    pickle_files = [os.path.join(base_dir, file) for file in os.listdir(base_dir)]
    pickle_files.sort()

    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
        for pickle_file in pickle_files:
            with open(pickle_file, 'rb') as file:
                video_data = pickle.load(file)
            # For each video file in the pickle file, write its contents to a new mp4 file
            for video_name, video_content in video_data.items():
                output_path = os.path.join(target_dir, f'{video_name}{suffix}')
                with open(output_path, 'wb') as output_file:
                    output_file.write(video_content)
        print('The video file has been restored and stored from the pickle file.')
    else:
        print('The video file already exists.')


def read_audio(audio_file):
    # 读取音频文件
    vr = decord.AudioReader(audio_file, ctx= decord.cpu(0), sample_rate=16000)
    return vr


def extract_segments(audio_file, sample_points, audio_segments_path, num_audio_seg):

    # 读取音频文件
    vr = read_audio(audio_file)
    vr = torch.from_numpy(vr._array)
    sample_rate = 16000
    
    # 每个片段的持续时间为1秒
    segment_duration = 1
    segment_samples = sample_rate * segment_duration
    valid_samples = []
    
    for i, point in enumerate(sample_points):
        for j in range(num_audio_seg):
            start_sample = int(point * sample_rate)
            end_sample = start_sample + segment_samples

            if end_sample > vr.shape[-1]:
                break

            samples = vr[:, start_sample:end_sample]
            torchaudio.save(audio_segments_path[i * num_audio_seg + j], samples, sample_rate)
            valid_samples.append(audio_segments_path[i * num_audio_seg + j])
    
    return valid_samples



class VideoMME(VideoBaseDataset):

    MD5 = '2f16cd40b1c125b67e661e59da2f6cd0'
    SYS = ''

    FRAMES_TMPL_NOSUB = ""

    FRAMES_TMPL_SUB = """
These are the frames of a video. \
This video's subtitles are listed below: \n
{}\n"""

    TYPE = 'MCQ'

    def __init__(self, dataset='Video-MME', use_subtitle=False):
        super().__init__(dataset=dataset)
        self.use_subtitle = use_subtitle
        self.num_audio_seg = 1

    @classmethod
    def supported_datasets(cls):
        return ['Video-MME']

    def prepare_dataset(self, dataset_name='Video-MME', repo_id='lmms-lab/Video-MME'):

        def check_integrity(pth):
            data_file = osp.join(pth, f'{dataset_name}.tsv')

            if not os.path.exists(data_file):
                return False

            if md5(data_file) != self.MD5:
                return False
            data = load(data_file)
            for video_pth in data['video_path']:
                if not osp.exists(osp.join(pth, video_pth)):
                    return False
            return True

        # cache_path = get_cache_path(repo_id)
        cache_path = '/mnt/data/shenghao/datasets/videomme'
        if cache_path is not None and check_integrity(cache_path):
            dataset_path = cache_path
        else:

            def unzip_hf_zip(pth):
                import zipfile
                base_dir = pth
                target_dir = os.path.join(pth, 'videos/')
                zip_files = [
                    os.path.join(base_dir, file) for file in os.listdir(base_dir)
                    if file.endswith('.zip') and file.startswith('video')
                ]
                zip_files.sort()

                if not os.path.exists(target_dir):
                    os.makedirs(target_dir, exist_ok=True)
                    for zip_file in zip_files:
                        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                            for member in zip_ref.namelist():
                                # Check if the member is a file (not a directory)
                                if not member.endswith('/'):
                                    # Extract the file to the specified directory
                                    source = zip_ref.open(member)
                                    target = open(os.path.join(target_dir, os.path.basename(member)), 'wb')
                                    with source, target:
                                        target.write(source.read())
                    print('The video file has been restored and stored from the zip file.')
                else:
                    print('The video file already exists.')

                subtitle_zip_file = os.path.join(base_dir, 'subtitle.zip')
                subtitle_target_dir = os.path.join(base_dir, 'subtitle')

                if not os.path.exists(subtitle_target_dir):
                    os.makedirs(subtitle_target_dir, exist_ok=True)
                    with zipfile.ZipFile(subtitle_zip_file, 'r') as zip_ref:
                        for member in zip_ref.namelist():
                            # Check if the member is a file (not a directory)
                            if not member.endswith('/'):
                                # Extract the file to the specified directory
                                source = zip_ref.open(member)
                                target = open(os.path.join(subtitle_target_dir, os.path.basename(member)), 'wb')
                                with source, target:
                                    target.write(source.read())
                    print('The subtitle file has been restored and stored from the zip file.')
                else:
                    print('The subtitle file already exists.')

            def generate_tsv(pth):

                data_file = osp.join(pth, f'{dataset_name}.tsv')
                if os.path.exists(data_file) and md5(data_file) == self.MD5:
                    return

                data_file = pd.read_parquet(os.path.join(pth, 'test-00000-of-00001.parquet'))
                data_file = data_file.assign(index=range(len(data_file)))
                data_file['video'] = data_file['videoID']
                data_file['video_path'] = data_file['videoID'].apply(lambda x: f'./videos/{x}.mp4')
                data_file['subtitle_path'] = data_file['videoID'].apply(lambda x: f'./subtitle/{x}.srt')
                data_file['question'] += '\n' + data_file['options'].apply(lambda x: '\n'.join(x))

                data_file = data_file[['index', 'video', 'video_path', 'duration', 'domain',
                                       'sub_category', 'task_type', 'subtitle_path', 'question', 'answer']]

                data_file.to_csv(osp.join(pth, f'{dataset_name}.tsv'), sep='\t', index=False)

            # dataset_path = snapshot_download(repo_id=repo_id, repo_type='dataset')
            dataset_path = '/mnt/data/shenghao/datasets/videomme'
            # unzip_hf_zip(dataset_path)
            generate_tsv(dataset_path)

        data_file = osp.join(dataset_path, f'{dataset_name}.tsv')

        return dict(data_file=data_file, root=dataset_path)

    def save_video_frames(self, video, num_frames=8):

        vid_path = osp.join(self.data_root, 'videos', video + '.mp4')
        try:
            vid = decord.VideoReader(vid_path)
        except:
            print('Error loading video:', vid_path)
            return [], [], [1], {'fps': 0, 'n_frames': 0,}
        step_size = len(vid) / (num_frames + 1)
        indices = [int(i * step_size) for i in range(1, num_frames + 1)]

        video_info = {
            'fps': vid.get_avg_fps(),
            'n_frames': len(vid),
        }

        frame_paths = self.frame_paths(video, num_frames)
        flag = np.all([osp.exists(p) for p in frame_paths])

        if not flag:
            images = [vid[i].asnumpy() for i in indices]
            images = [Image.fromarray(arr) for arr in images]
            for im, pth in zip(images, frame_paths):
                if not osp.exists(pth):
                    im.save(pth)

        try:
            audio_frame_timestamp = [idx / vid.get_avg_fps() for idx in indices]
            num_audio_seg = min(self.num_audio_seg, max((audio_frame_timestamp[1] - audio_frame_timestamp[0]) // 1, 1))
            audio_segments_path = []
            for audio_file in frame_paths:
                new_path = audio_file.replace('.jpg', '_%d.wav')
                for j in range(self.num_audio_seg):
                    audio_segments_path.append(new_path % j)

            audio_segments_path = extract_segments(vid_path, audio_frame_timestamp, audio_segments_path, num_audio_seg)
            return frame_paths, audio_segments_path, indices, video_info, num_audio_seg
        except:
            return frame_paths, [], indices, video_info, 0

    def build_prompt(self, line, num_frames, video_llm):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        frames, audios, indices, video_info, num_audio_seg = self.save_video_frames(line['video'], num_frames)

        # print(self.use_subtitle)
        # print(os.path.exists(osp.join(self.data_root, line['subtitle_path'])))
        # print(osp.join(self.data_root, line['subtitle_path']))
        if self.use_subtitle and os.path.exists(osp.join(self.data_root, line['subtitle_path'])):
            import pysubs2
            subs = pysubs2.load(osp.join(self.data_root, line['subtitle_path']), encoding='utf-8')
            subtitles = []

            for seleced_frame_id in indices:
                sub_text = ''
                cur_time = pysubs2.make_time(fps=video_info['fps'], frames=seleced_frame_id)
                for sub in subs:
                    if sub.start < cur_time and sub.end > cur_time:
                        sub_text = sub.text.replace('\\N', ' ')
                        break
                if sub_text.strip():
                    subtitles.append(sub_text)
            subtitles = '\n'.join(subtitles)
        else:
            subtitles = ''

        # message = [dict(type='text', value=self.SYS)]
        message = []
        if video_llm:
            message.append(dict(type='video', value=osp.join(self.data_root, 'videos', line['video'] + '.mp4')))
        else:
            total_audio_seg = 0
            for im in frames:
                message.append(dict(type='image', value=im))
                for j in range(num_audio_seg):
                    if total_audio_seg < len(audios):
                        message.append(dict(type='video_audio', value=audios[total_audio_seg]))
                        total_audio_seg += 1


        # print("=================================")
        # print(subtitles)
        # print("=================================")
        text_prompt = self.FRAMES_TMPL_NOSUB if not self.use_subtitle else self.FRAMES_TMPL_SUB.format(subtitles)
        prompt = "Question: {}\nAnswer with the option's letter from the given choices directly.".format(line['question'])
        text_prompt += prompt
        message.append(dict(type='text', value=text_prompt))
        return message

    # It returns a dictionary
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.videomme import get_dimension_rating, extract_characters_regex

        assert eval_file.endswith('.xlsx'), 'data file should be an xlsx file'

        tmp_file = eval_file.replace('.xlsx', '_tmp.pkl')
        tgt_file = eval_file.replace('.xlsx', '_rating.json')
        score_file = eval_file.replace('.xlsx', '_score.xlsx')

        if not osp.exists(score_file):
            res = {} if not osp.exists(tmp_file) else load(tmp_file)
            res = {k: v for k, v in res.items() if FAIL_MSG not in v}

            data = load(eval_file)
            data_un = data[~pd.isna(data['prediction'])]

            for idx in data['index']:
                ans = data.loc[data['index'] == idx, 'answer'].values[0]
                pred = data.loc[data['index'] == idx, 'prediction'].values[0]

                if extract_characters_regex(pred) == '':
                    data.loc[idx, 'score'] = -1
                else:
                    data.loc[idx, 'score'] = int(extract_characters_regex(pred) == ans)

            rejected = [x for x in data['score'] if x == -1]

            print(
                f'Among {len(data)} questions, failed to obtain prediction for {len(data) - len(data_un)} questions, '
                f'failed to obtain the score for another {len(rejected)} questions. '
                f'Those questions will be counted as -1 score in ALL rating, and will not be counted in VALID rating.'
            )

            dump(data, score_file)

        rating = get_dimension_rating(score_file)
        dump(rating, tgt_file)
        return rating

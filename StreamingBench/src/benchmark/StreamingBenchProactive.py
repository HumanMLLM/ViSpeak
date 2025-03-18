import tqdm
import os
import time
import json
from utils.data_execution import get_model_response
from utils.video_execution import split_video

from benchmark.Benchmark import Benchmark

PROMPT_TEMPLATE_PROACTIVE = '''You have been provided with images and a question related to the images. Your task is to carefully analyze the images and provide the answer to the question. You need to carefully confirm whether the images content meet the conditions of the question, and then output the correct content.
<video>\n
Question: {}

The answer is:
'''

class StreamingBenchProactive(Benchmark):
    def __init__(self, data, video_root):
        StreamingBenchProactiveInit(data)
        self.video_root = video_root

    def eval(self, data, model, output_path):
        StreamingBenchProactiveEval(data, model, output_path, self.video_root)

def StreamingBenchProactiveInit(data):
    pass

def StreamingBenchProactiveEval(data, MODEL, output_path, video_root):
    for subset in tqdm.tqdm(data):
        for question in subset["questions"]:
            if MODEL.name() in question and question[MODEL.name()]['dialog_history'][-1]['content']:
                continue

            video_path = subset["video_path"]
            timestamp = question["time_stamp"]
            ground_truth_timestamp = question["ground_truth_time_stamp"]

            # convert timestamps like "00:03:10" to seconds
            start_time = sum(int(x) * 60 ** i for i, x in enumerate(reversed(timestamp.split(":"))))
            ground_truth_time = sum(int(x) * 60 ** i for i, x in enumerate(reversed(ground_truth_timestamp.split(":"))))
            max_time = ground_truth_time + 4  # Maximum polling time: ground truth + 4 seconds

            dialog_history = []
            inp = question['question']
            time_s = time.time()
            file = os.path.join(video_root, video_path)
            response, response_time = get_model_response(MODEL, file, inp, start_time, max_time, start_time - 1, False, True)
            time_e = time.time()
            timecost = time_e - time_s

            # Record the interaction
            dialog_history.append({
                'role': 'assistant', 'content': response, 'time': response_time, 'cost': timecost
            })


            question[MODEL.name()] = {
                "dialog_history": dialog_history
            }

            with open(output_path, "w") as f:
                json.dump(data, f, indent=4)

from tqdm import tqdm
import os
import json
from utils.data_execution import get_model_response
from utils.video_execution import split_video

from benchmark.Benchmark import Benchmark

PROMPT_TEMPLATE = '''
Question: {}
Options:
{}
{}
{}
{}
Answer the question with only the letter (A, B, C, or D) of the correct option.'''

PROMPT_TEMPLATE_WITHOUT_OPTIONS = '''
Question: {}
Analyze the video and provide the answer to the question.
'''

class StreamingBenchOmni(Benchmark):
    def __init__(self, data, video_root):
        StreamingBenchInit(data)
        self.video_root = video_root

    def eval(self, data, model, output_path):
        StreamingBenchEval(data, model, output_path, self.video_root)

def StreamingBenchInit(data):
    pass

def StreamingBenchEval(data, MODEL, output_path, video_root):
    for subset in tqdm(data):
        for question in subset["questions"]:
            if MODEL.name() in question and question[MODEL.name()]:
                continue

            video_path = subset["video_path"]
            timestamp = question["time_stamp"]
            # convert timestamps like "00:03:10" to seconds
            timestamp = sum(int(x) * 60 ** i for i, x in enumerate(reversed(timestamp.split(":"))))

            file = os.path.join(video_root, video_path)

            ques = question["question"]
            if "options" in question.keys():
                options = question["options"]
                if not options[0].startswith("A."):
                    options = [f"A. {options[0]}", f"B. {options[1]}", f"C. {options[2]}", f"D. {options[3]}"]

                inp = PROMPT_TEMPLATE.format(ques, *options)
                inp += "\n\nThe best option is:"
            else:
                inp = PROMPT_TEMPLATE_WITHOUT_OPTIONS.format(ques)
                inp += "\n\nAnswer:"

            # print(f"input: {inp}")

            response, response_time = get_model_response(MODEL, file, inp, max(timestamp - 60, 0), timestamp, question_time=timestamp + 0.1, omni=True)
            question[MODEL.name()] = response

            with open(output_path, "w") as f:
                json.dump(data, f, indent=4)

            # remove the clip file
            # os.remove(file)
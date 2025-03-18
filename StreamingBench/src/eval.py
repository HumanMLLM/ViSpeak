from utils.data_execution import load_data

from model.modelclass import Model
from benchmark.Benchmark import Benchmark

import argparse

def main(args):
    data = load_data(args.data_file)

    ####### BENCHMARK #######

    benchmark = Benchmark(data)

    if args.benchmark_name == "Streaming":
        from benchmark.StreamingBench import StreamingBench
        benchmark = StreamingBench(data, args.video_root)
    if args.benchmark_name == "StreamingProactive":
        from benchmark.StreamingBenchProactive import StreamingBenchProactive
        benchmark = StreamingBenchProactive(data, args.video_root)
    if args.benchmark_name == "StreamingSQA":
        from benchmark.StreamingBenchSQA import StreamingBenchSQA
        benchmark = StreamingBenchSQA(data, args.video_root)
    if args.benchmark_name == "StreamingOmni":
        from benchmark.StreamingBenchOmni import StreamingBenchOmni
        benchmark = StreamingBenchOmni(data, args.video_root)

    ##########################

    ####### MODEL ############

    model = Model()
    from StreamingBench.src.model.ViSpeak import ViSpeak
    model = ViSpeak()

    ######################

    benchmark.eval(data, model, args.output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, required=True, help="Path to the data file")
    parser.add_argument("--video_root", type=str, required=True, help="Path to the video dictionary")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model")
    parser.add_argument("--benchmark_name", type=str, required=True, help="Name of the benchmark")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output file")
    args = parser.parse_args()
    main(args)
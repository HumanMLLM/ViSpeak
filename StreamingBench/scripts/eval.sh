cd ../src

# Change the model name to the model you want to evaluate

EVAL_MODEL="ViSpeak"
Devices=0

# For real-time visual understanding 
CUDA_VISIBLE_DEVICES=$Devices python eval.py --model_name $EVAL_MODEL --benchmark_name Streaming --data_file ./data/questions_real.json --output_file ./data/real_output_${EVAL_MODEL}.json --video_root /mnt/data1/shenghao/StreamingBench/


# For omni-source understanding
CUDA_VISIBLE_DEVICES=$Devices python eval.py --model_name $EVAL_MODEL --benchmark_name StreamingOmni --data_file ./data/questions_omni.json --output_file ./data/omni_output_${EVAL_MODEL}.json --video_root /mnt/data1/shenghao/StreamingBench/


# For sequential question answering
CUDA_VISIBLE_DEVICES=$Devices python eval.py --model_name $EVAL_MODEL --benchmark_name StreamingSQA --data_file ./data/questions_sqa.json --output_file ./data/sqa_output_${EVAL_MODEL}.json --video_root /mnt/data1/shenghao/StreamingBench/

# For proactive output
CUDA_VISIBLE_DEVICES=$Devices python eval.py --model_name $EVAL_MODEL --benchmark_name StreamingProactive --data_file ./data/questions_proactive.json --output_file ./data/proactive_output_${EVAL_MODEL}.json --video_root /mnt/data1/shenghao/StreamingBench/

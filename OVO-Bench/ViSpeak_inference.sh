python inference.py --anno_path /mnt/data/shenghao/datasets/OVO-Bench/ovo_bench.json --video_dir /mnt/data/shenghao/datasets/OVO-Bench/ --mode offline --model ViSpeak --model_path /mnt/data/shenghao/vita_v8/output/llava-s3-pretrain_video_merged

python score.py --model ViSpeak --mode offline
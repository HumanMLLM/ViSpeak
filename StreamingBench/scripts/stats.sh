cd ../src/data

# python count.py --model "<model_name>" --task "<real/omni/sqa/proactive>" --src "<output_file>"

python count.py --model "ViSpeak" --task "real" --src "real_output_ViSpeak.json"
python count.py --model "ViSpeak" --task "omni" --src "omni_output_ViSpeak.json"
python count.py --model "ViSpeak" --task "sqa" --src "sqa_output_ViSpeak.json"
python count.py --model "ViSpeak" --task "proactive" --src "proactive_output_ViSpeak.json"
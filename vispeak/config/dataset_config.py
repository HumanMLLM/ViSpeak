# stage 1 pretrain
text_dataset_0_25 = {"chat_path": "/mnt/data/shenghao/datasets/streaming_omni/text/Magpie-Llama-3.1-Pro-MT-300K-Filtered-slim_compact_v2.json", "data_ratio": 0.25}
image_dataset_0_25 = {"chat_path": "/mnt/data/shenghao/datasets/streaming_omni/llava_instruct_150k/sharegpt4v_mix665k_abspath_tts_compact_a100_v2.json", "data_ratio": 0.25}
video_dataset_0_25 = {"chat_path": "/mnt/data/shenghao/datasets/streaming_omni/LLaVA-Video-178K/llava_video_178k_1.3M_abspath_tts_a100.json", "data_ratio": 0.25}
audio_dataset_0_25 = {"chat_path": "/mnt/data/shenghao/datasets/streaming_omni/audio/new_audio_410k_compact_a100.json", "data_ratio": 0.25}

# stage 1 finetune
text_dataset = {"chat_path": "/mnt/data/shenghao/datasets/streaming_omni/text/Magpie-Llama-3.1-Pro-MT-300K-Filtered-slim_compact_v2.json", "data_ratio": 1}
image_dataset = {"chat_path": "/mnt/data/shenghao/datasets/streaming_omni/llava_instruct_150k/sharegpt4v_mix665k_abspath_tts_compact_a100_v2.json", "data_ratio": 1}
video_dataset = {"chat_path": "/mnt/data/shenghao/datasets/streaming_omni/LLaVA-Video-178K/llava_video_178k_1.3M_abspath_tts_a100.json", "data_ratio": 1}
audio_dataset = {"chat_path": "/mnt/data/shenghao/datasets/streaming_omni/audio/new_audio_410k_compact_a100.json", "data_ratio": 1}
cross_modality_data = {"chat_path": "/mnt/data/shenghao/datasets/streaming_omni/LLaVA-Video-178K/Ola_cross_modality_llava_121k.json", "data_ratio": 1}

# stage 2 finetune
offline_dataset = {"chat_path": "/mnt/data/shenghao/datasets/streaming_omni/streaming/omni_offline_370k_a100_v2.json", "data_ratio": 1, "streaming": False, "audio_folder_path": None, "video_folder_path": None, "image_folder_path": None}
offline_cross_modality_data = {"chat_path": "/mnt/data/shenghao/datasets/streaming_omni/LLaVA-Video-178K/Ola_cross_modality_llava_121k.json", "data_ratio": 1, "streaming": False, "audio_folder_path": None, "video_folder_path": None, "image_folder_path": None}
shot2story_magqa_dataset = {"chat_path": "/mnt/data/shenghao/datasets/streaming_omni/streaming/shot2story_magqa_37k.json", "data_ratio": 1, "streaming": True, "audio_folder_path": None, "video_folder_path": "/mnt/data/shenghao/datasets/shot2story/release_134k_videos/", "image_folder_path": None}
shot2story_dvc_dataset = {"chat_path": "/mnt/data/shenghao/datasets/streaming_omni/streaming/shot2story_dvc_37k.json", "data_ratio": 1, "streaming": True, "audio_folder_path": None, "video_folder_path": "/mnt/data/shenghao/datasets/shot2story/release_134k_videos/", "image_folder_path": None}
hirest_grounding_dataset = {"chat_path": "/mnt/data/shenghao/datasets/streaming_omni/streaming/hirest_grounding_0.5k.json", "data_ratio": 1, "streaming": True, "audio_folder_path": None, "video_folder_path": "/mnt/data/shenghao/datasets/HiREST/train/8a979260-c16b-4864-9184-d1134667840d/", "image_folder_path": None}
ET_Instruct_dataset = {"chat_path": "/mnt/data/shenghao/datasets/streaming_omni/streaming/ET_Instruct_42k.json", "data_ratio": 1, "streaming": True, "audio_folder_path": None, "video_folder_path": "/mnt/data/shenghao/datasets/ET-Instruct-164K/videos/", "image_folder_path": None}
EgoTimeQA_dataset = {"chat_path": "/mnt/data/shenghao/datasets/streaming_omni/streaming/EgoTimeQA_gqa_34k.json", "data_ratio": 1, "streaming": True, "audio_folder_path": None, "video_folder_path": "/mnt/data/shenghao/datasets/Ego4d/v2/clips/", "image_folder_path": None}
didemo_grounding_dataset = {"chat_path": "/mnt/data/shenghao/datasets/streaming_omni/streaming/didemo_grounding_6k.json", "data_ratio": 1, "streaming": True, "audio_folder_path": None, "video_folder_path": "/mnt/data/shenghao/datasets/DiDeMo_subset/", "image_folder_path": None}

ET_Instruct_proactive_dataset = {"chat_path": "/mnt/data/shenghao/datasets/streaming_omni/streaming/ET_Instruct_proactive.json", "data_ratio": 1, "streaming": True, "audio_folder_path": None, "video_folder_path": "/mnt/data1/shenghao/ET-Instruct-164K/videos/", "image_folder_path": None}


# stage 3 finetune
socialiqa_dataset = {"chat_path": "/mnt/data/shenghao/datasets/ViSpeak/ViSpeak-Instruct/social_i_qa_merged_4k.json", "data_ratio": 1, "streaming": False, "audio_folder_path": None, "video_folder_path": None, "image_folder_path": None}
socialiq_dataset = {"chat_path": "/mnt/data/shenghao/datasets/ViSpeak/ViSpeak-Instruct/socialiq_2K.json", "data_ratio": 1, "streaming": False, "audio_folder_path": None, "video_folder_path": "/mnt/data1/shenghao/social-iq/videos/", "image_folder_path": None}
intentqa_dataset = {"chat_path": "/mnt/data/shenghao/datasets/ViSpeak/ViSpeak-Instruct/intentqa_5k.json", "data_ratio": 1, "streaming": False, "audio_folder_path": None, "video_folder_path": "/mnt/data1/shenghao/IntentQA/", "image_folder_path": None}
smile_dataset = {"chat_path": "/mnt/data/shenghao/datasets/ViSpeak/ViSpeak-Instruct/SMILE_1k.json", "data_ratio": 1, "streaming": False, "audio_folder_path": None, "video_folder_path": "/mnt/data1/shenghao/SMILE/videos/video_clip/", "image_folder_path": None}

hivua_dataset = {"chat_path": "/mnt/data/shenghao/datasets/ViSpeak/ViSpeak-Instruct/Anomaly_Warning_HIVUA_train.json", "data_ratio": 1, "streaming": True, "audio_folder_path": None, "video_folder_path": "/mnt/data1/shenghao/HIVAU-70k/", "image_folder_path": None}
oops_dataset = {"chat_path": "/mnt/data/shenghao/datasets/ViSpeak/ViSpeak-Instruct/Anomaly_Warning_oops_train.json", "data_ratio": 1, "streaming": True, "audio_folder_path": None, "video_folder_path": "/mnt/data1/shenghao/UAL-Bench/", "image_folder_path": None}
funqa_dataset = {"chat_path": "/mnt/data/shenghao/datasets/ViSpeak/ViSpeak-Instruct/HumorQA_train_2k.json", "data_ratio": 1, "streaming": True, "audio_folder_path": None, "video_folder_path": "/mnt/data1/shenghao/FunQA/", "image_folder_path": None}

gesture_under_dataset = {"chat_path": "/mnt/data/shenghao/datasets/ViSpeak/ViSpeak-Instruct/socialiq_our_anno_1k.json", "data_ratio": 1, "streaming": True, "audio_folder_path": None, "video_folder_path": "/mnt/data1/shenghao/social-iq/", "image_folder_path": None}
gesture_pro_dataset = {"chat_path": "/mnt/data/shenghao/datasets/ViSpeak/ViSpeak-Instruct/Gesture_Understanding_trainv3_7k.json", "data_ratio": 1, "streaming": True, "audio_proactive": True, "audio_folder_path": None, "video_folder_path": "/mnt/data1/shenghao/self_collected_gesture/", "image_folder_path": None}
interrupt_text_dataset = {"chat_path": "/mnt/data/shenghao/datasets/ViSpeak/ViSpeak-Instruct/interruption_text_training_data.json", "data_ratio": 1, "streaming": True, "audio_proactive": True, "audio_folder_path": None, "video_folder_path": "/mnt/data1/shenghao/self_collected_data_en", "image_folder_path": None}
reference_text_dataset = {"chat_path": "/mnt/data/shenghao/datasets/ViSpeak/ViSpeak-Instruct/reference_text_training_data.json", "data_ratio": 1, "streaming": True, "audio_proactive": True, "no_visual_proactive": True, "audio_folder_path": None, "video_folder_path": "/mnt/data1/shenghao/self_collected_data_en", "image_folder_path": None}
wakeup_text_dataset = {"chat_path": "/mnt/data/shenghao/datasets/ViSpeak/ViSpeak-Instruct/wake_up_text_training_data_split.json", "data_ratio": 1, "streaming": True, "audio_proactive": True, "audio_folder_path": None, "video_folder_path": "/mnt/data1/shenghao/self_collected_data_en", "image_folder_path": None}



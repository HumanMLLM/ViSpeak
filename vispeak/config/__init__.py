from .dataset_config import *

offline_omni_data_0_25 = [text_dataset_0_25, image_dataset_0_25, video_dataset_0_25, audio_dataset_0_25]
offline_omni_data = [text_dataset, image_dataset, video_dataset, audio_dataset, cross_modality_data]
streaming_omni_data = [offline_dataset, offline_cross_modality_data, shot2story_magqa_dataset, shot2story_dvc_dataset, hirest_grounding_dataset, ET_Instruct_dataset, EgoTimeQA_dataset, didemo_grounding_dataset]
informative_data = [shot2story_magqa_dataset, shot2story_dvc_dataset, ET_Instruct_proactive_dataset]

ViSpeak_Instruct = [socialiqa_dataset, socialiq_dataset, intentqa_dataset, smile_dataset, hivua_dataset, oops_dataset, funqa_dataset, gesture_under_dataset, gesture_pro_dataset, interrupt_text_dataset, reference_text_dataset, wakeup_text_dataset]

ViSpeak_Instruct_no_offline = [hivua_dataset, oops_dataset, funqa_dataset, gesture_under_dataset, gesture_pro_dataset, interrupt_text_dataset, reference_text_dataset, wakeup_text_dataset]

DataConfig = {
    "pretrain_offline_omni_data": offline_omni_data_0_25,
    "fintune_offline_omni_data": offline_omni_data,
    "streaming_omni_data": streaming_omni_data,
    "informative_data": informative_data,
    "ViSpeak_Instruct": ViSpeak_Instruct,
    "ViSpeak_Instruct_no_offline": ViSpeak_Instruct_no_offline,
}


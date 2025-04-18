# Model Constants
MAX_IMAGE_LENGTH = 16  # 8#16#32#64
MIN_IMAGE_LENGTH = 4
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
AUDIO_TOKEN_INDEX = -500
VIDEO_AUDIO_TOKEN_INDEX = -600
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_AUDIO_TOKEN = "<audio>"
DEFAULT_VIDEO_AUDIO_TOKEN = "<video_audio>"
DEFAULT_SEG_TOKEN = ">Note"
DEFAULT_BOS_TOKEN = "<|startoftext|>"
CONTROLLER_HEART_BEAT_EXPIRATION = 30
LOGDIR = "gradio-logs"
WORKER_HEART_BEAT_INTERVAL = 15
DEFAULT_DATA_RATIO = 1.0#0.124#0.5 #0.2 #1.0
GLOBAL_WEIGHTS_PATH = "/path/to/model_weights"
DEFAULT_IMAGE_TOKEN_NUMBER = 256
DEFAULT_AUDIO_TOKEN_NUMBER = 12

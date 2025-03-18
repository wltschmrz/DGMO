import os
import yaml
from typing import Dict

def parse_yaml(config_yaml: str) -> Dict:
    with open(config_yaml, "r") as fr:
        return yaml.load(fr, Loader=yaml.FullLoader)

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def clean_wav_filenames(dir_path):
    if not os.path.exists(dir_path):
        return
    for filename in os.listdir(dir_path):
        if filename.endswith(".wav"):
            file_path = os.path.join(dir_path, filename)
            os.remove(file_path)
        elif filename.endswith(".png"):
            file_path = os.path.join(dir_path, filename)
            os.remove(file_path)

def ensure_folder_exists(folder_path):
    os.makedirs(folder_path, exist_ok=True)
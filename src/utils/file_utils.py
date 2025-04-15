import os
import yaml
from typing import Dict

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

def make_unique_dir(base_path: str, prefix: str = "single"):
    num = 1
    dir_path = os.path.join(base_path, f"{prefix}_{num}")
    while os.path.exists(dir_path):
        num += 1
        dir_path = os.path.join(base_path, f"{prefix}_{num}")
    os.makedirs(dir_path)
    return dir_path

def debug_wav(wav, label="wav"):
    print(f"\n--- {label} ---")
    print("type:", type(wav))
    if isinstance(wav, torch.Tensor):
        print("shape:", wav.shape)
        print("dtype:", wav.dtype)
        print("device:", wav.device)
        print("min/max:", wav.min().item(), "/", wav.max().item())
    elif isinstance(wav, np.ndarray):
        print("shape:", wav.shape)
        print("dtype:", wav.dtype)
        print("min/max:", np.min(wav), "/", np.max(wav))
    else:
        print("Unknown type")

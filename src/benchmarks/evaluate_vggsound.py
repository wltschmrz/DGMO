import os
import sys
import re
from typing import Dict, List
import traceback

proj_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
src_dir = os.path.join(proj_dir, 'src')
sys.path.extend([proj_dir, src_dir])

import matplotlib.pyplot as plt
import csv
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pathlib
import librosa
import yaml
os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")

from src.models.audioldm import AudioLDM
from src.models.audioldm2 import AudioLDM2
from src.models.auffusion import Auffusion
# from src.data_processing import AuffusionProcessor as AudioDataProcessor
from src.data_processing import AudioDataProcessor
from src.pipeline_auffusion import inference
import torchaudio
from utils import load_audio_torch, calculate_sisdr, calculate_sdr, get_mean_sdr_from_dict, parse_yaml

class AudioCapsEvaluator:
    def __init__(self, query='caption', sampling_rate=32000) -> None:

        self.query = query
        self.sampling_rate = sampling_rate
        with open(f'src/benchmarks/metadata/vggsound_eval.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            eval_list = [row for row in csv_reader][1:]
        self.eval_list = eval_list
        self.audio_dir = 'data/vggsound'

    def __call__(self, pl_model, config) -> Dict:
        print(f'Evaluation on AudioCaps with [{self.query}] queries.')
        
        processor, audioldm = pl_model
        device = audioldm.device

        for param in audioldm.parameters():
            param.requires_grad = False

        sisdrs_list = []
        sdris_list = []
        samples = config['samples']
        
        for eval_data in tqdm(self.eval_list[:samples]):

            # vggsound
            file_id, mix_wav, s0_wav, s0_text, s1_wav, s1_text = eval_data
            labels = s0_text

            mixture_path = os.path.join(self.audio_dir, mix_wav)
            source_path = os.path.join(self.audio_dir, s0_wav)

            text = [labels]

            config['text'] = text[0]
            caption = config['text']

            sisdr_li, sdri_li = inference(
                audioldm, 
                processor,
                target_path=source_path,
                mixed_path=mixture_path,
                config=config,
                file_id=file_id[-6:])

            sisdr_li.append(caption)
            sdri_li.append(caption)

            sisdrs_list.append(sisdr_li)
            sdris_list.append(sdri_li)
            
        sisdrs_array = np.array(sisdrs_list)  # (samples, iterations)
        sdris_array = np.array(sdris_list)    # (samples, iterations)

        return sisdrs_array, sdris_array

if __name__ == "__main__":
    from utils import ensure_folder_exists, clean_wav_filenames
    folders = ["./test/batch_samples", "./test/plot", "./test/result"]
    for folder in folders:
        ensure_folder_exists(folder)
        clean_wav_filenames(folder)

    eval = AudioCapsEvaluator(query='caption', sampling_rate=16000)

    audioldm = AudioLDM('cuda')
    device = audioldm.device
    processor = AudioDataProcessor(device=device)

    # for i in range(4, 5):
    config = {
        'num_epochs': 300,  # 50?
        'batchsize': 4,
        'strength': 0.7,  # 0.6,
        'learning_rate': 0.01,
        'iteration': 2,
        'samples': 50,  # number of samples to evaluate
        'steps': 25,  # 50
    }

    # mean_sisdr, mean_sdri = eval((processor, audioldm), config)
    sisdr_array, sdri_array = eval((processor, audioldm), config)

    def format_number(num):
        num = float(num)  # 문자열이 아니라 숫자로 변환
        formatted = f"{num:.4f}"
        return formatted if num < 0 else f" {formatted}"

    vec_format = np.vectorize(format_number)
    formatted_sisdrs = vec_format(sisdr_array[:, :2].astype(float))
    formatted_sdris = vec_format(sdri_array[:, :2].astype(float))

    combined_data = np.core.defchararray.add(formatted_sisdrs, ' / ')
    combined_data = np.core.defchararray.add(combined_data, formatted_sdris)

    df = pd.DataFrame(combined_data)
    df['caption'] = sisdr_array[:, 2].astype(str)  # caption 데이터는 sisdrs_array의 마지막 열 사용

    df.columns = [f'iter {i+1}' for i in range(2)] + ['caption']
    df.to_csv('sisdr_sdri_results_null.csv', index=False)
    print("CSV 저장 완료: sisdr_sdri_results.csv")

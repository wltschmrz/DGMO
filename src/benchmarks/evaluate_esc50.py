import os
import sys
from typing import Dict, List

proj_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
src_dir = os.path.join(proj_dir, 'src')
sys.path.extend([proj_dir, src_dir])

import matplotlib.pyplot as plt

import csv
import pandas as pd
import numpy as np
from tqdm import tqdm
import torchaudio
import pathlib
import librosa
import yaml
os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")

import soundfile as sf
from src.models.audioldm import AudioLDM
from src.models.audioldm2 import AudioLDM2
from src.models.auffusion import Auffusion
from src.data_processing import AuffusionProcessor as AudioDataProcessor
from src.data_processing import AudioDataProcessor as prcssr
from src.pipeline_auffusion import inference
from utils import (
    load_audio_torch,
    get_mean_sdr_from_dict,
    parse_yaml,
)

class ESC50Evaluator:
    def __init__(self, sampling_rate=32000) -> None:

        self.sampling_rate = sampling_rate

        with open(f'src/benchmarks/metadata/esc50_eval.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            eval_list = [row for row in csv_reader][1:]
        self.eval_list = eval_list
        self.audio_dir = 'data/esc50'

    def __call__(self, pipeline, config) -> Dict:
        print(f'Evaluation on ESC-50 with [text label] queries.')
        
        processor, audioldm = pipeline
        device = audioldm.device

        for param in audioldm.parameters():
            param.requires_grad = False

        sisdrs_list = []
        sdris_list = []
        samples = config['samples']
        
        for eval_data in tqdm(self.eval_list[samples:samples+30]):

            idx, caption, _, _ = eval_data

            source_path = os.path.join(self.audio_dir, f'segment-{idx}.wav')
            mixture_path = os.path.join(self.audio_dir, f'mixture-{idx}.wav')

            text = [caption]
            config['text'] = text[0]

            sisdr_li, sdri_li = inference(audioldm, processor,
                      target_path=source_path,
                      mixed_path=mixture_path,
                      config=config,
                      file_id=idx)

            sisdr_li.append(caption)
            sdri_li.append(caption)

            sisdrs_list.append(sisdr_li)
            sdris_list.append(sdri_li)
            
        sisdrs_array = np.array(sisdrs_list)  # (samples, iterations)
        sdris_array = np.array(sdris_list)    # (samples, iterations)

        return sisdrs_array, sdris_array

if __name__ == "__main__":
    from utils import ensure_folder_exists, clean_wav_filenames
    
    # Ensure the folders exist before calling clean_wav_filenames
    folders = ["./test/batch_samples", "./test/plot", "./test/result"]

    for folder in folders:
        ensure_folder_exists(folder)
        clean_wav_filenames(folder)

    eval = ESC50Evaluator(sampling_rate=16000)
    
    audioldm = Auffusion('cuda')
    device = audioldm.device
    processor = AudioDataProcessor(device=device)

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
    formatted_sisdrs = vec_format(sisdr_array[:, :config['iteration']].astype(float))
    formatted_sdris = vec_format(sdri_array[:, :config['iteration']].astype(float))

    combined_data = np.char.add(formatted_sisdrs, ' | ')
    combined_data = np.char.add(combined_data, formatted_sdris)

    df = pd.DataFrame(combined_data)
    df['caption'] = sisdr_array[:, config['iteration']].astype(str)  # caption 데이터는 sisdrs_array의 마지막 열 사용

    df.columns = [f'iter {i+1}' for i in range(config['iteration'])] + ['caption']
    df.to_csv('sisdr_sdri_results.csv', index=False)
    print("CSV 저장 완료: sisdr_sdri_results.csv")

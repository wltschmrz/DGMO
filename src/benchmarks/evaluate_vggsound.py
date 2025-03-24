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
import torchaudio
from tqdm import tqdm
import pathlib
import librosa
import yaml
os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")

from src.utils import read_wav_file, calculate_sisdr, calculate_sdr, get_mean_sdr_from_dict
from pipeline_new import DGMO

class VGGSoundEvaluator:
    def __init__(
            self,
            metadata_pth='./src/benchmarks/metadata/vggsound_eval.csv',
            audio_dir='./data/vggsound',
            ) -> None:

        with open(metadata_pth) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            eval_list = [row for row in csv_reader][1:]
        self.eval_list = eval_list
        self.audio_dir = audio_dir

    def __call__(self, model, sample_num, **kwargs) -> Dict:
        print(f'Evaluation on VGGSound.')

        sdris_list = []
        sisdrs_list = []
        
        for eval_data in tqdm(self.eval_list):

            file_id, mix_wav, s0_wav, s0_text, s1_wav, s1_text = eval_data
            labels = s0_text

            mixture_path = os.path.join(self.audio_dir, mix_wav)
            source_path = os.path.join(self.audio_dir, s0_wav)

            text = [labels]

            # print(mixture_path)
            sep_wav = model.inference(
                mix_wav_path=mixture_path,
                text=text[0],
                save_path=f"./test/vgg_result/{file_id}.wav",
                )

            gt_wav = read_wav_file(filename=source_path, target_duration=10.24, target_sr=16000)

            sdr = calculate_sdr(gt_wav, sep_wav)
            sisdr = calculate_sisdr(gt_wav, sep_wav)

            sdris_list.append(sdr)
            sisdrs_list.append(sisdr)
            
        return sdris_list, sisdrs_list

if __name__ == "__main__":
    from utils import ensure_folder_exists, clean_wav_filenames
    folders = ["./test/vgg_result"]
    for folder in folders:
        ensure_folder_exists(folder)
        clean_wav_filenames(folder)

    eval = VGGSoundEvaluator()

    model = DGMO(config_path="./configs/DGMO.yaml", device="cuda:1")

    # mean_sisdr, mean_sdri = eval((processor, audioldm), config)
    sdris_list, sisdrs_list = eval(model, sample_num=-1)

    df = pd.DataFrame(zip(sdris_list, sisdrs_list))
    df.to_csv("./output.csv", index=False, header=False, encoding="utf-8")
    print("CSV 저장 완료: sisdr_sdri_results.csv")

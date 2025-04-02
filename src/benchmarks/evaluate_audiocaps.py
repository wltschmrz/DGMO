import os
import sys
from typing import Dict

proj_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
src_dir = os.path.join(proj_dir, 'src')
sys.path.extend([proj_dir, src_dir])

import csv
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
# os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")
from src.utils import read_wav_file, printing_sdrs, get_mean_sdr_from_dict
from src.utils import ensure_folder_exists, clean_wav_filenames, plot_wav_mel
from src.pipeline import DGMO

class AudioCapsEvaluator:
    def __init__(
            self,
            query='caption',
            metadata_pth=f'./src/benchmarks/metadata/audiocaps_eval.csv',
            audio_dir=f'./data/audiocaps',
            ) -> None:
        
        self.query = query
        with open(metadata_pth) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            eval_list = [row for row in csv_reader][1:]
        self.eval_list = eval_list
        self.audio_dir = audio_dir

    def __call__(self, model, config=None, mode="plain", **kwargs) -> Dict:
        print(f'Evaluation on AudioCaps with [{self.query}] queries.')
        assert mode in ["plain", "joint"], "check mode"

        result_dir = f"./test/caps_results"
        clean_wav_filenames(result_dir)
        ensure_folder_exists(result_dir)

        sdrs_li = []
        sisdrs_li = []
        sdris_li = []
        sisdris_li = []
        try:
            for eval_data in tqdm(self.eval_list):

                idx, caption, labels, src_wav, noise_wav = eval_data

                mixture_path = os.path.join(self.audio_dir, f'mixture-{idx}.wav')
                source_path = os.path.join(self.audio_dir, f'segment-{idx}.wav')
                                
                if self.query == 'caption':
                    text = [caption]
                elif self.query == 'labels':
                    text = [labels]

                if mode=="plain":
                    est_wav = model.inference(
                        mix_wav_path=mixture_path,
                        text=text[0],
                        save_path=f"./test/caps_results/{idx}_pl_sep.wav",
                        )
                elif mode=="joint":
                    print("Warning: Not Implemented Yet")

                mixed_wav = read_wav_file(filename=mixture_path, target_duration=10.24, target_sr=16000)
                ref_wav = read_wav_file(filename=source_path, target_duration=10.24, target_sr=16000)
                
                scores = printing_sdrs(ref=ref_wav, mix=mixed_wav, est=est_wav, printing=False)
                plot_wav_mel([mixed_wav, est_wav, ref_wav], idx=idx,
                            save_path=f"./test/caps_results/{idx}_mel_{mode}.png",
                            score=scores, config_path=config, text=text[0])

                sdr, sisdr, sdri, sisdri = scores
                sdrs_li.append(sdr)
                sisdrs_li.append(sisdr)
                sdris_li.append(sdri)
                sisdris_li.append(sisdri)
        except:
            pass
        finally:    
            return sdrs_li, sisdrs_li, sdris_li, sisdris_li
        
if __name__ == "__main__":
    from utils import clean_wav_filenames, ensure_folder_exists
    folders = ["./test/caps_results"]
    for folder in folders:
        ensure_folder_exists(folder)
        clean_wav_filenames(folder) ##

    config = "./configs/DGMO.yaml"
    eval = AudioCapsEvaluator(query='caption')
    model = DGMO(config_path=config, device="cuda:3")  ##

    sdrs_li, sisdrs_li, sdris_li, sisdris_li = eval(model, config, "plain")  ##

    df = pd.DataFrame(zip(sdrs_li, sisdrs_li, sdris_li, sisdris_li))
    df.to_csv("./test/caps_plain.csv", index=False, header=False, encoding="utf-8")  ##
    print("CSV 저장 완료: ./test/caps_plain.csv")  ##
    import numpy as np
    mean_sdr = np.mean(sdrs_li)
    mean_sisdr = np.mean(sisdrs_li)
    mean_sdri = np.mean(sdris_li)
    mean_sisdri = np.mean(sisdris_li)
    print(f"\n>> SDR: {mean_sdr}\n>> SISDR: {mean_sisdr}\n\
>> SDRi: {mean_sdri}\n>> SISDRi: {mean_sisdri}"
)

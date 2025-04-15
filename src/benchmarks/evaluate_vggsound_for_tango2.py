import os
import sys
from typing import Dict

proj_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
src_dir = os.path.join(proj_dir, 'src')
sys.path.extend([proj_dir, src_dir])
src2_dir = os.path.join(proj_dir, 'tango')
sys.path.extend([proj_dir, src_dir, src2_dir])
import csv
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
# os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")
from src.utils import read_wav_file, printing_sdrs, get_mean_sdr_from_dict
from src.utils import ensure_folder_exists, clean_wav_filenames, plot_wav_mel
from src.pipeline import DGMO
from contextlib import redirect_stdout

def plot_mel_raw(mel_raw, save_path="./mel_raw.png"):
    def preprocess(mel):
        if isinstance(mel, torch.Tensor):
            mel = mel.detach().cpu().numpy()
        mel = np.squeeze(mel)
        if mel.ndim == 3:
            mel = mel[0]
        if mel.shape[0] > mel.shape[1]:  # (T, F) → (F, T)
            mel = mel.T
        return mel

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    names = ["Original", "Reconstructed"]

    mel = preprocess(mel_raw)
    ax.imshow(mel, aspect='auto', origin='lower', cmap='inferno')
    ax.set_title(f"{names[1]} Mel")
    ax.set_xlabel("Time")
    ax.set_ylabel("Mel bins")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()

class VGGSoundEvaluator:
    def __init__(
            self,
            metadata_pth=f'./src/benchmarks/metadata/audiocaps_eval.csv',
            audio_dir=f'./data/audiocaps',
            ) -> None:

        with open(metadata_pth) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            eval_list = [row for row in csv_reader][1:]
        self.eval_list = eval_list
        self.audio_dir = audio_dir

    def __call__(self, model, config=None, mode="plain", **kwargs) -> Dict:
        print(f'Evaluation on VGGSound.')
        assert mode in ["plain", "joint"], "check mode"

        result_dir = f"./test/caps_results"
        clean_wav_filenames(result_dir)
        ensure_folder_exists(result_dir)

        sdrs_li = []
        sisdrs_li = []
        sdris_li = []
        sisdris_li = []

        sdrs_li2 = []
        sisdrs_li2 = []
        sdris_li2 = []
        sisdris_li2 = []
        try:
            for eval_data in tqdm(self.eval_list[:100]):
                # if i < 16:
                #     continue

                idx, caption, labels, src_wav, noise_wav = eval_data

                mixture_path = os.path.join(self.audio_dir, f'mixture-{idx}.wav')
                source_path = os.path.join(self.audio_dir, f'segment-{idx}.wav')

                text = [caption]

                if mode=="plain":
                    with open(os.devnull, 'w') as f, redirect_stdout(f):
                        est_wav1, ref_mel1 = model.inference(
                            mix_wav_path=mixture_path,
                            text=text[0],
                            save_dir=f"{result_dir}/{idx}",
                            save_fname=f"pl_sep.wav",  ##
                            thresholding=False,
                            )
                    
                    # est_wav2 = model.inference(
                    #     mix_wav_path=mixture_path,
                    #     text=texts[1],
                    #     save_dir=f"./test/vgg_results/{file_id}",
                    #     save_fname=f"pl_sep_1.wav"
                    #     )

                # elif mode=="joint":
                #     est_wav1, est_wav2 = model.joint_opt_inference(
                #         mix_wav_path=mixture_path,
                #         text=texts,
                #         save_dir=f"./test/vgg_results/{file_id}",
                #         )

                mixed_wav = read_wav_file(filename=mixture_path, target_duration=10.24, target_sr=16000)
                ref_wav1 = read_wav_file(filename=source_path, target_duration=10.24, target_sr=16000)
                # ref_wav2 = read_wav_file(filename=source_path2, target_duration=10.24, target_sr=16000)

                scores1 = printing_sdrs(ref=ref_wav1, mix=mixed_wav, est=est_wav1, printing=False)
                plot_wav_mel([mixed_wav, est_wav1, ref_wav1], idx=idx,
                            save_path=f"{result_dir}/{idx}/mel_{mode}_src_iter1.png",
                            score=scores1, config_path=config, text=text[0])
                
                plot_mel_raw(ref_mel1, save_path=f"{result_dir}/{idx}/mel_{mode}_ref_iter1.png",)
                # scores2 = printing_sdrs(ref=ref_wav2, mix=mixed_wav, est=est_wav2, printing=False)
                # plot_wav_mel([mixed_wav, est_wav2, ref_wav2], idx=file_id,
                #             save_path=f"./test/vgg_results/{file_id}/mel_{mode}_1.png",
                #             score=scores2, config_path=config, text=texts[1])

                sdr, sisdr, sdri, sisdri = scores1
                sdrs_li.append(sdr)
                sisdrs_li.append(sisdr)
                sdris_li.append(sdri)
                sisdris_li.append(sisdri)

                # sdr2, sisdr2, sdri2, sisdri2 = scores2
                # sdrs_li2.append(sdr2)
                # sisdrs_li2.append(sisdr2)
                # sdris_li2.append(sdri2)
                # sisdris_li2.append(sisdri2)
        except:
            pass
        finally:    
            return (sdrs_li, sisdrs_li, sdris_li, sisdris_li), #(sdrs_li2, sisdrs_li2, sdris_li2, sisdris_li2)

if __name__ == "__main__":
    from utils import ensure_folder_exists, clean_wav_filenames
    folders = ["./test/caps_results"]
    for folder in folders:
        ensure_folder_exists(folder)
        # clean_wav_filenames(folder)  ##

    config = "./configs/DGMO.yaml"
    eval = VGGSoundEvaluator()
    model = DGMO(config_path=config, device="cuda:1")  ##

    # mean_sisdr, mean_sdri = eval((processor, audioldm), config)
    sets = eval(model, config, "plain")  ##

    for i, setss in enumerate(sets):
        sdrs_li, sisdrs_li, sdris_li, sisdris_li = setss
        df = pd.DataFrame(zip(sdrs_li, sisdrs_li, sdris_li, sisdris_li))
        df.to_csv(f"./test/caps_plain{i}_iter1.csv", index=False, header=False, encoding="utf-8")  ##
        print(f"CSV 저장 완료: ./test/caps_plain{i}_iter1.csv")  ##
        import numpy as np
        mean_sdr = np.mean(sdrs_li)
        mean_sisdr = np.mean(sisdrs_li)
        mean_sdri = np.mean(sdris_li)
        mean_sisdri = np.mean(sisdris_li)
        print(f"\n>> SDR: {mean_sdr}\n>> SISDR: {mean_sisdr}\n\
>> SDRi: {mean_sdri}\n>> SISDRi: {mean_sisdri}"
    )

# -JUBdOr8Hes_000030+++37Vxf0Wz-3o_000001,data/mix_wav/-JUBdOr8Hes_000030+++37Vxf0Wz-3o_000001.wav,data/s0_wav/-JUBdOr8Hes_000030+++37Vxf0Wz-3o_000001.wav,playing accordion,data/s1_wav/-JUBdOr8Hes_000030+++37Vxf0Wz-3o_000001.wav,sea lion barking

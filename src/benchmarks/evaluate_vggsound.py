import os
import sys
from typing import Dict

proj_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
src_dir = os.path.join(proj_dir, 'src')
sys.path.extend([proj_dir, src_dir])
src2_dir = os.path.join(proj_dir, 'tango')
sys.path.extend([proj_dir, src_dir, src2_dir])
import csv
import pandas as pd
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
# os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")
from src.utils import read_wav_file, printing_sdrs, get_mean_sdr_from_dict
from src.utils import ensure_folder_exists, clean_wav_filenames, plot_wav_mel
from src.pipeline import DGMO

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

    def __call__(self, model, config=None, mode="plain", **kwargs) -> Dict:
        print(f'Evaluation on VGGSound.')
        assert mode in ["plain", "joint"], "check mode"

        result_dir = f"./test/vgg_results"
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
            for eval_data in tqdm(self.eval_list):
                # if i < 16:
                #     continue

                file_id, mix_wav, s0_wav, s0_text, s1_wav, s1_text = eval_data

                mixture_path = os.path.join(self.audio_dir, mix_wav)
                source_path1 = os.path.join(self.audio_dir, s0_wav)
                source_path2 = os.path.join(self.audio_dir, s1_wav)

                texts = [s0_text, s1_text]

                if mode=="plain":
                    est_wav1 = model.inference(
                        mix_wav_path=mixture_path,
                        text=texts[0],
                        save_dir=f"./test/vgg_results/{file_id}",
                        save_fname=f"pl_sep_0_3.wav"
                        )
                    
                    # est_wav2 = model.inference(
                    #     mix_wav_path=mixture_path,
                    #     text=texts[1],
                    #     save_dir=f"./test/vgg_results/{file_id}",
                    #     save_fname=f"pl_sep_1.wav"
                    #     )

                elif mode=="joint":
                    est_wav1, est_wav2 = model.joint_opt_inference(
                        mix_wav_path=mixture_path,
                        text=texts,
                        save_dir=f"./test/vgg_results/{file_id}",
                        )

                mixed_wav = read_wav_file(filename=mixture_path, target_duration=10.24, target_sr=16000)
                ref_wav1 = read_wav_file(filename=source_path1, target_duration=10.24, target_sr=16000)
                # ref_wav2 = read_wav_file(filename=source_path2, target_duration=10.24, target_sr=16000)

                scores1 = printing_sdrs(ref=ref_wav1, mix=mixed_wav, est=est_wav1, printing=False)
                plot_wav_mel([mixed_wav, est_wav1, ref_wav1], idx=file_id,
                            save_path=f"./test/vgg_results/{file_id}/mel_{mode}_0_3_manyiter.png",
                            score=scores1, config_path=config, text=texts[0])
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
    folders = ["./test/vgg_results"]
    for folder in folders:
        ensure_folder_exists(folder)
        # clean_wav_filenames(folder)  ##

    config = "./configs/DGMO.yaml"
    eval = VGGSoundEvaluator()
    model = DGMO(config_path=config, device="cuda:3")  ##

    # mean_sisdr, mean_sdri = eval((processor, audioldm), config)
    sets = eval(model, config, "plain")  ##

    for i, setss in enumerate(sets):
        sdrs_li, sisdrs_li, sdris_li, sisdris_li = setss
        df = pd.DataFrame(zip(sdrs_li, sisdrs_li, sdris_li, sisdris_li))
        df.to_csv(f"./test/vgg_plain{i}_3.csv", index=False, header=False, encoding="utf-8")  ##
        print(f"CSV 저장 완료: ./test/vgg_plain{i}_3.csv")  ##
        import numpy as np
        mean_sdr = np.mean(sdrs_li)
        mean_sisdr = np.mean(sisdrs_li)
        mean_sdri = np.mean(sdris_li)
        mean_sisdri = np.mean(sisdris_li)
        print(f"\n>> SDR: {mean_sdr}\n>> SISDR: {mean_sisdr}\n\
>> SDRi: {mean_sdri}\n>> SISDRi: {mean_sisdri}"
    )

# -JUBdOr8Hes_000030+++37Vxf0Wz-3o_000001,data/mix_wav/-JUBdOr8Hes_000030+++37Vxf0Wz-3o_000001.wav,data/s0_wav/-JUBdOr8Hes_000030+++37Vxf0Wz-3o_000001.wav,playing accordion,data/s1_wav/-JUBdOr8Hes_000030+++37Vxf0Wz-3o_000001.wav,sea lion barking

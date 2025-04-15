import os
import csv
from tqdm import tqdm
import torchaudio
import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from librosa.filters import mel as librosa_mel_fn
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display as ld
import os
import yaml
from typing import Dict

def segment_wav(waveform, target_len, start=0):  # [1,N+] → [1,N]
    sample_length = waveform.shape[-1]
    assert sample_length > 100, f"Waveform is too short, # of samples: {sample_length}"
    if sample_length <= target_len:  # too short
        return waveform
    elif sample_length > target_len:  # segmentation
        return waveform[:, start : start + target_len]

def pad_wav(waveform, target_len):  # [1,N-] → [1,N]
    sample_length = waveform.shape[-1]
    assert sample_length > 100, f"Waveform is too short, # of samples: {sample_length}"
    if sample_length == target_len:  # if same length
        return waveform
    elif sample_length < target_len:  # padding
        padded_wav = torch.zeros((1, target_len))
        padded_wav[:, :sample_length] = waveform
        return padded_wav

# fname → wav
def read_wav_file(filename, target_duration, target_sr):  # fname → np[1,N]
    # 1. file load
    wav, ori_sr = torchaudio.load(filename, normalize=True)  # ts[C,N'±]
    # 2. to mono channel
    wav = wav.mean(dim=0) if wav.shape[0] > 1 else wav  # ts[1,N'±]
    # 2. segment & padding (to target length)
    target_t = int(ori_sr * target_duration)
    wav = segment_wav(wav, target_t)  # ts[1,N'-]
    wav = pad_wav(wav, target_t)      # ts[1,N']
    # 3. resampling
    wav = torchaudio.functional.resample(wav, ori_sr, target_sr)  # ts[1,N]
    return wav.numpy()  # np[1,N]

def get_mean_sdr_from_dict(sdris_dict):
    mean_sdr = np.nanmean(list(sdris_dict.values()))
    return mean_sdr

def calculate_sdr(ref: np.ndarray, est: np.ndarray, eps=1e-10) -> float:
    r"""Calculate SDR between reference and estimation.
    Args:
        ref (np.ndarray), reference signal
        est (np.ndarray), estimated signal
    """
    reference = ref
    noise = est - reference
    numerator = np.clip(a=np.mean(reference ** 2), a_min=eps, a_max=None)
    denominator = np.clip(a=np.mean(noise ** 2), a_min=eps, a_max=None)
    sdr = 10. * np.log10(numerator / denominator)
    return sdr

def calculate_sisdr(ref, est):
    r"""Calculate SDR between reference and estimation.
    Args:
        ref (np.ndarray), reference signal
        est (np.ndarray), estimated signal
    """
    eps = np.finfo(ref.dtype).eps
    reference = ref.copy()
    estimate = est.copy()
    reference = reference.reshape(reference.size, 1)
    estimate = estimate.reshape(estimate.size, 1)
    Rss = np.dot(reference.T, reference)
    # get the scaling factor for clean sources
    a = (eps + np.dot(reference.T, estimate)) / (Rss + eps)
    e_true = a * reference
    e_res = estimate - e_true
    Sss = (e_true**2).sum()
    Snn = (e_res**2).sum()
    sisdr = 10 * np.log10((eps+ Sss)/(eps + Snn))
    return sisdr

def calculate_sdri(ref: np.ndarray, mix: np.ndarray, est: np.ndarray, eps=1e-10) -> float:
    r"""Calculate SDRi between reference and estimation.
    Args:
        ref (np.ndarray), reference signal
        mix (np.ndarray), mixture signal
        est (np.ndarray), estimated signal
    """
    prev_sdr = calculate_sdr(ref, mix, eps)
    improv_sdr = calculate_sdr(ref, est, eps)
    return improv_sdr - prev_sdr

def calculate_sisdri(ref: np.ndarray, mix: np.ndarray, est: np.ndarray) -> float:
    r"""Calculate SDRi between reference and estimation.
    Args:
        ref (np.ndarray), reference signal
        mix (np.ndarray), mixture signal
        est (np.ndarray), estimated signal
    """
    prev_sisdr = calculate_sisdr(ref, mix)
    improv_sisdr = calculate_sisdr(ref, est)
    return improv_sisdr - prev_sisdr

def printing_sdrs(*, ref, mix, est, printing=True, mode="all", eps=1e-10):
    '''
    Args:
        mode, is one of ["all", "basic", "improv"]
    '''
    sdr = calculate_sdr(ref, est)
    sisdr = calculate_sisdr(ref, est)
    sdri = calculate_sdri(ref, mix, est)
    sisdri = calculate_sisdri(ref, mix, est)

    return (sdr, sisdr, sdri, sisdri)

def plot_wav_mel(
        wav_arrays, 
        sr=16000, 
        save_path="./mel.png", 
        idx=None,
        score=(0,0), 
        config_path=None,
        text=None,
        **kwargs
        ):
    fig, axes = plt.subplots(2, len(wav_arrays), figsize=(4 * len(wav_arrays)+3, 6.24))
    clip_duration = 10.24  # 클리핑 길이 (초)
    hop_length = 512       # Hop length 설정

    for i, wav in enumerate(wav_arrays):
        if i==0:
            name = "Mix"
        elif i == 1:
            name = "Est"
        elif i == 2:
            name = "Ref"
        if len(wav.shape) > 1:
            wav = wav.squeeze()
        duration = len(wav) / sr
        if duration > clip_duration:
            wav = wav[: int(clip_duration * sr)]  # 앞 10.24초만 유지
        time = np.linspace(0, len(wav) / sr, num=len(wav))

        axes[0, i].plot(time, wav, lw=0.5)
        axes[0, i].set_title(f"{name} Waveform")
        axes[0, i].set_xlabel("Time (s)")
        axes[0, i].set_ylabel("Amplitude")
        axes[0, i].set_ylim([-0.5, 0.5])

        mel_spec = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=128, hop_length=hop_length)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        ld.specshow(
            mel_spec_db,
            sr=sr,
            hop_length=hop_length,
            x_axis="time",
            y_axis="mel",
            vmin=-80,
            vmax=0,
            ax=axes[1, i]
        )
        axes[1, i].set_title(f"{name} Mel Spectrogram")
    
    # ▶ 제목은 ID만 표시
    plt.suptitle(f"ID: {idx} / Target: {text}", fontsize=16)

    # ▶ SDR 정보는 바로 아래 따로 줄 생성
    if len(score) == 2:
        score_text = f"SDR: {score[0]:.2f}\
    SISDR: {score[1]:.2f}"
    elif len(score) == 4:
        score_text = f"SDR: {score[0]:.2f}\
    SISDR: {score[1]:.2f}\
    SDRi: {score[2]:.2f}\
    SISDRi: {score[3]:.2f}"
    else:
        score_text = ""
    fig.text(0.5, 0.91, score_text, fontsize=14, ha='center')

    # ▶ config는 맨 아래 여백에 출력
    if config_path is not None:
        config = load_config(config_path)
        config.update(kwargs)
        config_strs = [f"{k}: {v}" for k, v in config.items()]
        if "audioldm2" in config_strs[0]:
            config_strs[0] = "ldm: AudioLDM2"
        elif "audioldm" in config_strs[0]:
            config_strs[0] = "ldm: AudioLDM"
        elif "auffusion" in config_strs[0]:
            config_strs[0] = "ldm: Auffusion"
        config_text = "\n".join(config_strs)
    else:
        config_text = ""
    plt.subplots_adjust(right=0.85)
    fig.text(0.853, 0.35, config_text, fontsize=12, va='top', ha='left', 
             linespacing=1.4, fontfamily='monospace',
             bbox=dict(
        facecolor='white',   # 박스 배경색
        edgecolor='gray',    # 박스 테두리색
        boxstyle='round,pad=0.4',  # 둥근 박스 + padding
        linewidth=1.0
    ))
    plt.tight_layout(rect=[0, 0, 0.84, 0.96])  # 위 12%, 아래 12% 여백 확보
    plt.savefig(save_path, dpi=300)
    plt.close()

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)



if __name__ == "__main__":
    metadata_pth='../DGMO-Separation/src/benchmarks/metadata/vggsound_eval.csv'  ##
    audio_dir='../DGMO-Separation/data/vggsound'  ##
                # metadata_pth='./src/benchmarks/metadata/vggsound_eval.csv',
        # audio_dir='../DGMO-Separation/data/vgg'

    with open(metadata_pth) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        eval_list = [row for row in csv_reader][1:]

    sdrs = {
            '0': 0,
            '1': 0,
            '2': 0,
            '3': 0, 
        }

    for i, eval_data in enumerate(tqdm(eval_list)):
        file_id, mix_wav, s0_wav, s0_text, s1_wav, s1_text = eval_data

        mixture_path = os.path.join(audio_dir, mix_wav)
        source_path = os.path.join(audio_dir, s0_wav)
        sep_pth = f"./output-languageTSE/{file_id}_pred.wav"

        wav_mix = read_wav_file(filename=mixture_path, target_duration=10.24, target_sr=16000)
        wav_sep = read_wav_file(filename=sep_pth, target_duration=10.24, target_sr=16000)
        wav_gt = read_wav_file(filename=source_path, target_duration=10.24, target_sr=16000)

        scores = printing_sdrs(ref=wav_gt, mix=wav_mix, est=wav_sep, printing=False)
        
        if i % 50 == 0:
            wav_paths = [wav_mix, wav_sep, wav_gt]

            plot_wav_mel(
                wav_paths,
                save_path=f"./output-languageTSE/vgg_mels/{file_id}.png",
                score=scores,
                idx=file_id,
                config_path=None
            )

        scores = list(scores)
        scores = [float(s) for s in scores]

        for i, score in enumerate(scores):
            sdrs[str(i)] += score

    for k, v in sdrs.items():
        sdrs[k] = v / len(eval_list)
        print(f"SDR{k}: {sdrs[k]:.4f}")

    # SDR: 2.3841
    # SI-SDR: -8.6105
    # SDRi: 1.9108
    # SI-SDRi: -9.2961
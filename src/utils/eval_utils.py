import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display as ld
from .file_utils import load_config

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
    if printing:
        match mode:
            case "all":
                print(f"SDR: {sdr:.4f}, SI-SDR: {sisdr:.4f}")
                print(f"SDRi: {sdri:.4f}, SI-SDRi: {sisdri:.4f}")
            case "basic":
                print(f"SDR: {sdr:.4f}, SI-SDR: {sisdr:.4f}")
            case "improv":
                print(f"SDRi: {sdri:.4f}, SI-SDRi: {sisdri:.4f}")
            case _:
                raise ValueError
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

        if not np.issubdtype(wav.dtype, np.floating):
            wav = wav.astype(np.float32) / np.iinfo(wav.dtype).max

        duration = len(wav) / sr
        if duration > clip_duration:
            wav = wav[: int(clip_duration * sr)]  # 앞 10.24초만 유지
        
        time = np.linspace(0, len(wav) / sr, num=len(wav))
        axes[0, i].plot(time, wav, lw=0.5)
        
        axes[0, i].set_title(f"{name} Waveform")
        axes[0, i].set_xlabel("Time (s)")
        axes[0, i].set_ylabel("Amplitude")
        axes[0, i].set_ylim([-1, 1])
        
        mel_spec = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=128, hop_length=hop_length)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        mel_spec_db = np.squeeze(mel_spec_db)
        assert mel_spec_db.ndim == 2, f"mel_spec_db must be 2D, got shape {mel_spec_db.shape}"

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
    plt.subplots_adjust(right=0.85)
    fig.text(0.853, 0.4, config_text, fontsize=12, va='top', ha='left', 
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
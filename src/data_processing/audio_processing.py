import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import numpy as np
import torch
import torchaudio
from librosa.filters import mel as librosa_mel_fn
from src.utils.file_utils import load_config

def segment_wav(waveform, target_length, start=0):  # [1,N+] → [1,N]
    sample_length = waveform.shape[-1]
    assert sample_length > 100, f"Waveform is too short, # of samples: {sample_length}"
    if sample_length <= target_length:  # too short
        return waveform
    elif sample_length > target_length:  # segmentation
        return waveform[:, start : start + target_length]

def pad_wav(waveform, target_length):  # [1,N-] → [1,N]
    sample_length = waveform.shape[-1]
    assert sample_length > 100, f"Waveform is too short, # of samples: {sample_length}"
    if sample_length == target_length:  # if same length
        return waveform
    elif sample_length < target_length:  # padding
        padded_wav = torch.zeros((1, target_length))
        padded_wav[:, :sample_length] = waveform
        return padded_wav

# fname → wav
def read_wav_file(filename, duration, target_sr):  # fname → np[1,N]
    # 1. file load
    wav, ori_sr = torchaudio.load(filename, normalize=True)  # ts[C,N'±]
    # 2. to mono channel
    wav = wav.mean(dim=0) if wav.shape[0] > 1 else wav  # ts[1,N'±]
    # 2. segment & padding (to target length)
    target_t = int(ori_sr * duration)
    wav = segment_wav(wav, target_t)  # ts[1,N'-]
    wav = pad_wav(wav, target_t)      # ts[1,N']
    # 3. resampling
    wav = torchaudio.functional.resample(wav, ori_sr, target_sr)  # ts[1,N]
    return wav.numpy()  # np[1,N]

def save_wav_file(filename, wav_np, target_sr):
    wav = torch.tensor(wav_np, dtype=torch.float32)
    if len(wav.shape) == 1:  # [N] → [1, N] (mono 채널)
        wav = wav.unsqueeze(0)
    wav = (wav * 32767).clamp(-32768, 32767).short()
    torchaudio.save(filename, wav, target_sr, encoding="PCM_S", bits_per_sample=16)
    print(f"Saved WAV file: {filename} (Sample Rate: {target_sr} Hz)")

class AudioDataProcessor():
    def __init__(self, *, config_path=None, device=None, **kwargs):
        self.device = device

        # wav normalizing params
        self.norm_shifting = None
        self.wav_max = None
        # spec reconstruction params
        self.spec_length = None

        # STFT params
        self.mel_basis = {}
        self.hann_window = {}

        config = load_config(config_path) if config_path else {}
        config.update(kwargs)
        self._apply_config(config)

        self.n_freq = self.filter_length // 2 + 1  # F: 513
        self.sample_length = self.sampling_rate * self.duration  # N: 163840
        self.pad_size = int((self.filter_length - self.hop_length) / 2)  # (1024-160)/2 = 432
        self.n_times = int(((self.sample_length + 2 * self.pad_size) 
                            - self.win_length) // self.hop_length +1)  # 1024

        if (device not in self.mel_basis or
            device not in self.hann_window):
            
            mel_filterbank = librosa_mel_fn(  # np[M:64, F:513]
                sr=self.sampling_rate,
                n_fft=self.filter_length,
                n_mels=self.mel_bins,
                fmin=self.mel_fmin,
                fmax=self.mel_fmax,
                )
            
            self.mel_basis[f"{device}"] = torch.from_numpy(mel_filterbank).float().to(device)  # ts[M,F:513]
            self.hann_window[f"{device}"] = torch.hann_window(self.win_length).to(device)  # ts[win_length] = [1024]

        if self.repo_id == "cvssp/audioldm":
            assert self.n_freq == 513, f"n_freq should be 513, but {self.n_freq}"
            assert self.sample_length == 163840, f"sample_length should be 163840, but {self.sample_length}"
            assert self.pad_size == 432, f"pad_size should be 432, but {self.pad_size}"
            assert self.n_times == 1024, f"n_times should be 1024, but {self.n_times}"
            print("[INFO] audio_processing.py: set for AudioLDM")
        elif self.repo_id == "cvssp/audioldm2":
            assert self.n_freq == 513, f"n_freq should be 513, but {self.n_freq}"
            assert self.sample_length == 163840, f"sample_length should be 163840, but {self.sample_length}"
            assert self.pad_size == 432, f"pad_size should be 432, but {self.pad_size}"
            assert self.n_times == 1024, f"n_times should be 1024, but {self.n_times}"
            print("[INFO] audio_processing.py: set for AudioLDM2")
        elif self.repo_id == "auffusion/auffusion":
            assert self.n_freq == 513, f"n_freq should be 513, but {self.n_freq}"
            assert self.sample_length == 163840, f"sample_length should be 163840, but {self.sample_length}"
            assert self.pad_size == 432, f"pad_size should be 432, but {self.pad_size}"
            assert self.n_times == 1024, f"n_times should be 1024, but {self.n_times}"
            print("[INFO] audio_processing.py: set for Auffusion")

    def _apply_config(self, config):
        for key, value in config.items():
            if isinstance(value, dict):
                self._apply_config(value)
            else:
                setattr(self, key, value)

    def normalize_wav(self, waveform):  # [1,N] → [1,N]
        MAX_AMPLITUDE = 0.5
        EPSILON = 1e-8
        assert self.norm_shifting == self.wav_max == None, "Normalization params should not be set"
        self.norm_shifting = np.mean(waveform)
        centered = waveform - self.norm_shifting
        self.wav_max = np.max(np.abs(centered))
        normalized = centered * MAX_AMPLITUDE / (self.wav_max + EPSILON)
        return normalized    # in [-0.5,0.5]

    def denormalize_wav(self, normed_wav, factor_removal=True):  # [1,N] → [1,N]
        AMPLITUDED = 2
        EPSILON = 1e-8
        centered = normed_wav * AMPLITUDED * (self.wav_max + EPSILON)
        origin_wav = centered + self.norm_shifting
        if factor_removal:
            self.norm_shifting = None
            self.wav_max = None
        return origin_wav

    def spectral_normalize_torch(self, magnitudes, C=1, CLIP_VAL=1e-5):  # dynamic_range_compression
        return torch.log(torch.clamp(magnitudes, min=CLIP_VAL) * C)

    # fname → wav
    def read_wav_file(self, filename):  # fname → np[1,N]
        # 1. file load
        wav, ori_sr = torchaudio.load(filename, normalize=True)  # ts[C,N'±]
        # 2. to mono channel
        wav = wav.mean(dim=0) if wav.shape[0] > 1 else wav  # ts[1,N'±]
        # 2. segment & padding (to target length)
        target_t = int(ori_sr * self.duration)
        wav = segment_wav(wav, target_t)  # ts[1,N'-]
        wav = pad_wav(wav, target_t)      # ts[1,N']
        # 3. resampling
        wav = torchaudio.functional.resample(wav, ori_sr, self.sampling_rate)  # ts[1,N]
        return wav.numpy()  # np[1,N]
    
    # wav → wav' (LDM input용)
    def prepare_wav(self, wav):  # np[1,N] → np[1,N]
        # 4. normalize
        wav = self.normalize_wav(wav)  # centering & Norm [-0.5,0.5]
        wav = torch.FloatTensor(wav)
        return wav  # ts[1,N]
    
    # wav' → stft_mag, stft_complex
    def wav_to_stft(self, waveform):  # ts[1,N] → ts[1, F:513, T:1024]
        waveform = waveform.to(self.device)
        assert torch.min(waveform) >= -1, f"train min value is {torch.min(waveform)}"
        assert torch.max(waveform) <= 1, f"train max value is {torch.max(waveform)}"

        # # waveform: np[1,N] → [1,1,N] → [1,1, N+2*pad_size] → [1, N+2*pad_size] = ts[1, 164704]
        # waveform = torch.nn.functional.pad(waveform.unsqueeze(1), (self.pad_size, self.pad_size), mode="reflect").squeeze(1)

        stft_complex = torch.stft(  # ts[1, F:513, T:1024~30] (complex)
            waveform,                   # F = filter_length // 2 + 1 (onesided=True) = 513
            self.filter_length,         # T = ((samples + 2*pad_size) - win_length) // hop_length + 1 = 1024
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.hann_window[f"{self.device}"],
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        stft_mag = torch.abs(stft_complex)  # ts[1, F:513, T:1024~30]
        self.spec_length = stft_mag.shape[-1]
        stft_mag, stft_complex = stft_mag[:,:,:self.target_length], stft_complex[:,:,:self.target_length]  # ts[F,T], ts[F,T]
        assert stft_mag.shape[:2] == (1, self.n_freq), f"{stft_mag.shape}, {self.n_freq}, {self.n_times}"
        return stft_mag, stft_complex  # [1, F:513, T:1024], [1, F:513, T:1024]
    
    # stft_mag → mel_spec
    def stft_to_mel(self, stft_mag):  # ts[1, F:513, T:1024] → ts[1, M:64, T:1024]
        mel_filterbank = self.mel_basis[f"{self.device}"]  # ts[M:64, F:513]
        # [M:64, F:513] x [1, F:513, T:1024] → [1, M:64, T:1024]
        stft_mag = stft_mag.to(self.device)  # ts[1, F:513, T:1024]
        mel_spec = self.spectral_normalize_torch(torch.matmul(mel_filterbank, stft_mag))  # ts[1, M:64, T:1024]
        assert mel_spec.shape[1:] == (self.mel_bins, self.target_length), f"{mel_spec.shape}, {stft_mag.shape}"
        return mel_spec  # ts[1, M:64, T:1024]
    
    # mel_spec → vae_input
    def preprocess_spec(self, spectrogram):  # ts[1, M, T] -> ts[1, 1, T*:1024, M*:64]
        spec = spectrogram[0]  # [M, T]
        spec = spec.T.float()  # [T, M]
        n_frames = spec.shape[0]
        p = self.target_length - n_frames
        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            spec = m(spec)  # [T*, M] 뒷 시간 늘림
        elif p < 0:
            spec = spec[0 : self.target_length, :]  # [T*, M] 뒷 시간 줄임
        if (spec.size(-1) % 2 != 0):
            spec = spec[..., :-1]  # M이 odd면, -1  # [T*, M*]
        return spec[None, None, ...]  # [1, 1, T*, M*]
    
    # stft → wav' → wav
    def inverse_stft(self, stft_mag, stft_complex, fac_rm=True):  # ts[1,F,T], ts[1,F,T] → ts[1,N]
        assert stft_mag.shape == stft_complex.shape
        assert stft_mag.shape[1:] == (self.n_freq, self.n_times), f"{stft_mag.shape}"
        if stft_mag.shape[-1] < self.spec_length:
            _stft_mag = torch.zeros((1, self.n_freq, self.spec_length), device=self.device)
            _stft_mag[:, :, :self.n_times] = stft_mag
            _stft_complex = torch.zeros((1, self.n_freq, self.spec_length), dtype=torch.complex64, device=self.device)
            _stft_complex[:, :, :self.n_times] = stft_complex
        else:
            _stft_mag = stft_mag[:, :, :self.spec_length]
            _stft_complex = stft_complex[:, :, :self.spec_length]
        assert _stft_mag.shape[1:] == (self.n_freq, self.spec_length), f"{_stft_mag.shape}"
        stft_mag = _stft_mag.squeeze(0)
        stft_complex = _stft_complex.squeeze(0)

        eps=1e-5
        phase = stft_complex / (stft_complex.abs() + eps)
        masked_stft_complex = stft_mag * phase

        estimated_wav = torch.istft(
                masked_stft_complex,
                n_fft=self.filter_length,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.hann_window[f"{self.device}"],
                normalized=False,
                onesided=True
            )
        
        # # forward에서 reflect pad를 (pad_size, pad_size)만큼 했었으므로 앞뒤로 pad_size samples씩 잘라낸다.
        # pad_size = self.pad_size
        # if estimated_wav.shape[-1] > pad_size*2:
        #     estimated_wav = estimated_wav[..., pad_size:-pad_size]  # shape [B, samples]
        # else:
        #     # 혹시 길이가 매우 짧다면 예외처리
        #     estimated_wav = estimated_wav[..., 0:1]
        wav = self.reconst_wav(estimated_wav.unsqueeze(0), factor_rm=fac_rm)  # ts[1,N] → np[1,N]
        return wav  # np[1,N]
    
    # wav' → wav
    def reconst_wav(self, wav, factor_rm=True):  # ts[1,N] → np[1,N]
        wav = wav.detach().cpu().numpy()
        assert self.norm_shifting is not None and self.wav_max is not None, "Normalization params should be set"
        assert wav.shape[0] == 1, f"Waveform shape is not [1,N], but {wav.shape}"
        wav = self.denormalize_wav(wav, factor_removal=factor_rm)  # UN centering & normalizing
        return wav  # np[1,N]



if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from src.utils.eval_utils import calculate_sdr, calculate_sisdr
    
    config_path = "configs/audioldm.yaml"
    audio_processor = AudioDataProcessor(config_path=config_path, device="cuda")
    ori_wav = audio_processor.read_wav_file('data/samples/A_cat_meowing.wav')  # np[1,N]
    
    sdr = calculate_sdr(ori_wav, ori_wav)
    sisdr = calculate_sisdr(ori_wav, ori_wav)

    print(">> Original: \n", sdr, sisdr)
    
    wav = audio_processor.prepare_wav(ori_wav)  # ts[1,N]
    _wav = audio_processor.reconst_wav(wav, factor_rm=False)  # np[1,N]
    assert ori_wav.shape == _wav.shape, f"{ori_wav.shape}, {_wav.shape}"
    assert ori_wav.dtype == _wav.dtype, f"{ori_wav.dtype}, {_wav.dtype}"

    sdr_ = calculate_sdr(ori_wav, _wav)
    sisdr_ = calculate_sisdr(ori_wav, _wav)

    print(">> Norm & Denorm: \n", sdr_, sisdr_)

    stft_mag, stft_complex = audio_processor.wav_to_stft(wav)  # ts[1,F,T], ts[1,F,T]
    mel_spec = audio_processor.stft_to_mel(stft_mag)  # ts[1,M,T]
    vae_input = audio_processor.preprocess_spec(mel_spec)  # ts[1,1,T*,M*]
    __wav = audio_processor.inverse_stft(stft_mag, stft_complex, fac_rm=True)  # np[1,N]
    assert ori_wav.shape == __wav.shape, f"{ori_wav.shape}, {__wav.shape}"
    assert ori_wav.dtype == __wav.dtype, f"{ori_wav.dtype}, {__wav.dtype}"

    sdr__ = calculate_sdr(ori_wav, __wav)
    sisdr__ = calculate_sisdr(ori_wav, __wav)

    print(">> Norm + stft & istft + Denorm: \n", sdr__, sisdr__)

    # --- #

    import numpy as np

    filename = 'data/samples/A_cat_meowing.wav'
    output_filename = "test/test.wav"
    target_sr = 16000  # 16kHz 샘플링 레이트

    wav_np_original = read_wav_file(filename, duration=10.24, target_sr=target_sr)
    save_wav_file(output_filename, wav_np_original, target_sr)
    wav_np_reloaded = read_wav_file(output_filename, duration=10.24, target_sr=target_sr)

    are_equal = np.allclose(wav_np_original, wav_np_reloaded, atol=1e-4)
    print("데이터 동일 여부:", are_equal)
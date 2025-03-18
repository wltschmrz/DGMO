import os
import numpy as np
import torch
import torchaudio
from librosa.filters import mel as librosa_mel_fn
import os
import random
import numpy as np
import torch
import torch.nn.functional as F

class AuffusionProcessor():
    def __init__(self, device="cuda", config=None):
        self.device = device

        self.norm_shifting = 0
        self.wav_max = 0

        self.do_trim_wav = False
        self.waveform_only = False
        self.do_random_segment = False

        self.sampling_rate = 16000
        self.duration = config.get('duration', 10.24) if config else 10.24
        self.target_length = 1024
        self.mixup = 0.0

        self.mel_basis = {}
        self.hann_window = {}

        # DSP: s-full 기준 (audioldm_original.yaml)
        self.filter_length = 2048  # n_fft
        self.hop_length = 160
        self.win_length = 1024
        self.n_mel = 256  # M: 64
        self.mel_fmin = 0
        self.mel_fmax = 8000

        self.n_freq = self.filter_length // 2 + 1  # F: 513
        self.sample_length = self.sampling_rate * self.duration  # N: 163840
        self.pad_size = int((self.win_length - self.hop_length) / 2)  # (1024-160)/2 = 432 # auffusion 944
        self.n_times = int(((self.sample_length + 2 * self.pad_size) - self.win_length) // self.hop_length +1)  # 123 #  affusion 1030

    # --------------------------------------------------------------------------------------------- #

    def segment_wav(self, waveform, target_length, start=0):  # [1,N+] → [1,N]
        waveform_length = waveform.shape[-1]
        assert waveform_length > 100, f"Waveform is too short, {waveform_length}"
        # Too short
        if waveform_length <= target_length:
            return waveform
        segment = waveform[:, start:start + target_length]
        return segment

    def pad_wav(self, waveform, target_length):  # [1,N-] → [1,N]
        waveform_length = waveform.shape[-1]
        assert waveform_length > 100, f"Waveform is too short, {waveform_length}"
        # if same length
        if waveform_length == target_length:
            return waveform
        # padding (target이 더 긴때만 처리하면 됨)
        padded_wav = torch.zeros((1, target_length))
        padded_wav[:, :waveform_length] = waveform
        return padded_wav
    

    
    def denormalize_wav(self, normed_wav):  # [1,N] → [1,N]
        AMPLITUDED = 2
        EPSILON = 1e-8
        centered = normed_wav * AMPLITUDED * (self.wav_max + EPSILON)
        origin_wav = centered + self.norm_shifting
        return origin_wav



    ### fname → wav' ###  (LDM input용)
    def prepare_wav_from_filename(self, filename):  # fname → np[1,N]
        wav = self.read_wav_file(filename)  # np[1,N]
        # 4. normalize
        wav = self.normalize_wav(wav)  # centering & Norm [-0.5,0.5]
        wav = torch.FloatTensor(wav)
        return wav  # ts[1,N]


    ### wav' → wav ###
    def reconst_wav(self, wav):  # ts[1,N] → np[1,N]
        wav = wav.detach().cpu().numpy()
        wav = self.denormalize_wav(wav)  # UN centering & Norm
        return wav  # np[1,N]

    # --------------------------------------------------------------------------------------------- #

    ### 범용 용도 wav ###
    def read_wav_file(self, filename):  # fname → np[1,N]
        # 1. file load
        wav, raw_sr = torchaudio.load(filename, normalize=True)  # ts[C,N'±]
        # 2. to mono channel
        wav = wav.mean(dim=0) if wav.shape[0] > 1 else wav  # ts[1,N'±]
        # 2. segment & padding (to target length)
        raw_length = int(raw_sr * self.duration)
        wav = self.segment_wav(wav, raw_length)  # ts[1,N'-]
        wav = self.pad_wav(wav, raw_length)      # ts[1,N']
        # 3. resampling
        wav = torchaudio.functional.resample(wav, raw_sr, self.sampling_rate)  # ts[1,N]
        return wav.numpy()  # np[1,N]

     ### wav → wav' ###  (LDM input용)
    def prepare_wav(self, wav):  # np[1,N] → np[1,N]
        # 4. normalize
        wav = self.normalize_wav(wav)  # centering & Norm [-0.5,0.5]
        wav = torch.FloatTensor(wav)
        return wav  # ts[1,N]

    def normalize_wav(self, waveform):  # [1,N] → [1,N]
        MAX_AMPLITUDE = 0.5
        EPSILON = 1e-8
        self.norm_shifting = np.mean(waveform)
        centered = waveform - self.norm_shifting
        self.wav_max = np.max(np.abs(centered))
        normalized = centered * MAX_AMPLITUDE / (self.wav_max + EPSILON)
        return normalized    # in [-0.5,0.5]

    ### wav' → input mel ###  auffusion을 위해 수정 중
    def wav_to_mel(self, wav):  # wav: ts[C,N] → logmel: ts[1,1,T,M] / stft: ts[1,1,T,F]
        wav = wav.mean(dim=0) if wav.shape[0] > 1 else wav  # ts[1,N]
        waveform = torch.FloatTensor(wav).to(self.device)  # ts[1,N]
        stft, stft_c = self.wav_to_stft(waveform)  # ts[1, F:513, T:1024] / ts[1, F:513, T:1024]
        log_mel_spec = self.stft_to_mel(stft)  # ts[1, M:64, T:1024]

        #log_mel_spec, p = self.postprocess_spec(log_mel_spec)  # ts[T:1024, M:64]
        norm_spec = affusion_normalize_spectrogram(log_mel_spec)
        norm_spec = pad_spec(norm_spec, 1024)
        # norm_spec = normalize(norm_spec)
        
        return norm_spec  # ts[1,1,T,M]


    ### wav' → stft_mag, stft_complex ###
    def wav_to_stft(self, waveform):  # ts[1,N] → ts[1, F:513, T:1024]
        waveform = waveform.to(self.device)
        assert torch.min(waveform) >= -1, f"train min value is {torch.min(waveform)}"
        assert torch.max(waveform) <= 1, f"train min value is {torch.max(waveform)}"

        if self.mel_fmax not in self.mel_basis:
            mel_filterbank = librosa_mel_fn(  # np[M:64, F:513]
                sr=self.sampling_rate,
                n_fft=self.filter_length,
                n_mels=self.n_mel,
                fmin=self.mel_fmin,
                fmax=self.mel_fmax,)
            
            self.mel_basis[f"{self.mel_fmax}_{self.device}"] = torch.from_numpy(mel_filterbank).float().to(self.device)  # ts[M,F:513]
            self.hann_window[f"{self.device}"] = torch.hann_window(self.win_length).to(self.device)  # ts[win_length] = [1024]

        # ========== wav -> stft ==========
        # waveform: np[1,N] → [1,1,N] → [1,1, N+2*pad_size] → [C, N+2*pad_size] = ts[C, 164704]
        waveform = torch.nn.functional.pad(waveform.unsqueeze(1), (self.pad_size, self.pad_size), mode="reflect").squeeze(1)

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
        stft_mag, stft_complex = stft_mag[:,:,:self.target_length], stft_complex[:,:,:self.target_length]  # ts[F,T], ts[F,T]
        assert stft_complex.shape == stft_mag.shape
        assert stft_mag.shape[:2] == (1,self.n_freq), f"{stft_mag.shape}, {self.n_freq}, {self.n_times}"
        return stft_mag, stft_complex  # [1, F:513, T:1024], [1, F:513, T:1024]

    def stft_to_mel(self, stft_mag):  # ts[1, F:513, T:1024] → ts[1, M:64, T:1024]
        # ========== stft -> mel ==========
        mel_filterbank = self.mel_basis[f"{self.mel_fmax}_{self.device}"]  # ts[M:64, F:513]
        # [M:64, F:513] x [1, F:513, T:1024] → [1, M:64, T:1024]
        stft_mag = stft_mag.to(self.device)  # ts[1, F:513, T:1024]
        mel_spec = self.spectral_normalize_torch(torch.matmul(mel_filterbank, stft_mag))  # ts[1, M:64, T:1024]

        assert mel_spec.shape[1] == self.n_mel, f"{mel_spec.shape}, {stft_mag.shape}"
        return mel_spec  # ts[1, M:64, T:1024]
    
    def spectral_normalize_torch(self, magnitudes, C=1, CLIP_VAL=1e-5):  # dynamic_range_compression
        return torch.log(torch.clamp(magnitudes, min=CLIP_VAL) * C)

    def postprocess_spec(self, spectrogram):  # [1, ~, T] -> [T*, ~*]
        spec = spectrogram[0]  # [~, T]
        spec = spec.T.float()  # [T, ~]
        spec, p = self.pad_spec(spec)  # [T*, ~*]
        return spec[None, None, ...], p


    
    def pad_spec(self, spectrogram):  # [T, ~] → [T*, ~*]
        n_frames = spectrogram.shape[0]
        p = self.target_length - n_frames
        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            spectrogram = m(spectrogram)  # [T*, ~] 뒷 시간 늘림
        elif p < 0:
            spectrogram = spectrogram[0 : self.target_length, :]  # [T*, ~] 뒷 시간 줄임
        if (spectrogram.size(-1) % 2 != 0):
            spectrogram = spectrogram[..., :-1]  # ~ 가 odd면, -1
        return spectrogram, p


    def reversing_stft(self, stft):
        if len(stft.shape) == 3:
            stft = stft.squeeze(0)
        assert stft.shape == torch.Size([self.n_times, self.n_freq]), f"{stft.shape}"
        stft = stft.T.float()
        stft = stft.unsqueeze(0)
        return stft


    
    ### masked stft → masked mel ###
    def masked_stft_to_masked_mel(self, stft_mag):  # ts[1, F:513, T:1024] → ts[1, M:64, T:1024]
        mel_spec = self.stft_to_mel(stft_mag)  # ts[1, M:64, T:1024]
        log_mel_spec, p = self.postprocess_spec(mel_spec)  # ts[T:1024, M:64]
        return log_mel_spec  # ts[1,1,T,M]
    
    # --------------------------------------------------------------------------------------------- #

    ### fname → mel ###
    def read_audio_file(self, filename, pad_stft=False):  # → ts[t,mel], ts[t,freq], ts[C,samples]
        # 1. 오디오 파일 로드 또는 빈 파형 생성
        if os.path.exists(filename):
            waveform = self.prepare_wav_from_filename(filename)  # np[C,N], int
        else:
            target_length = int(self.sampling_rate * self.duration)
            waveform = torch.zeros((1, target_length)), 0  # np[C,samples], int
            print(f'Non-fatal Warning [dataset.py]: The wav path "{filename}" not found. Using empty waveform.')

        # 2. 특성 추출 (stft spec, log mel spec)
        log_mel_spec = None if self.waveform_only else self.wav_to_mel(waveform)  # input: [1,N]
        return log_mel_spec  # ts[1,1,T,M]

    ### mel → wav ###
    def inverse_mel_with_phase(
        self,
        masked_mel_spec: torch.Tensor,    # 모델이 예측한 log mel spec, shape [B, 1, T, M]
        stft_complex: torch.Tensor,       # forward에서 구한 복소 STFT, shape [B, F, T]
        eps=1e-5
    ):
        # 1) mel_spec (log scale) → linear scale로 변환
        assert masked_mel_spec.shape == (1,1,1024,256)
        masked_mel_spec = masked_mel_spec.squeeze(0)
        masked_mel_linear = torch.exp(masked_mel_spec)  # shape [1, T, M]

        # 2) mel → STFT magnitude로 근사* 복원
        mel_filterbank = self.mel_basis[f"{self.mel_fmax}_{self.device}"]
        inv_mel_filter = torch.pinverse(mel_filterbank)  # shape [F, M]

        # 현재 masked_mel_linear: [B, T, n_mel] → [B, n_mel, T] 로 transpose
        masked_mel_linear = masked_mel_linear.permute(0, 2, 1)  # shape [B, M, T]

        # pseudo-inverse 곱: [F, M] x [M, T] = [F, T]
        # 배치처리까지 고려, map으로 처리
        batch_size = masked_mel_linear.shape[0]
        masked_stft_mag = []
        for i in range(batch_size):
            # shape [M, T] → [F, T]
            mag_i = inv_mel_filter @ masked_mel_linear[i]
            masked_stft_mag.append(mag_i.unsqueeze(0))
        masked_stft_mag = torch.cat(masked_stft_mag, dim=0)  # shape [B, F, T]

        # 3) 원본 stft_complex의 phase 추출 후, magnitude와 결합
        # phase = stft_complex / (|stft_complex| + eps)
        phase = stft_complex / (stft_complex.abs() + eps)
        masked_stft_complex = masked_stft_mag * phase  # shape 동일: [B, F, T]

        # 4) iSTFT 수행 (forward와 동일 파라미터)
        hann_window = self.hann_window[f"{self.device}"]

        estimated_wav = torch.istft(
            masked_stft_complex.to(self.device),
            n_fft=self.filter_length,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=hann_window,
            normalized=False,
            onesided=True
        )  # shape [B, samples + 2*pad_size]

        # 5) pad_size 부분 제거
        #    forward에서 reflect pad를 (pad_size, pad_size)만큼 했었으므로,
        #    최종 waveform에서 앞뒤로 pad_size samples씩 잘라낸다.
        if estimated_wav.shape[-1] > self.pad_size*2:
            estimated_wav = estimated_wav[..., self.pad_size:-self.pad_size]  # shape [B, samples]
        else:
            # 혹시 길이가 매우 짧다면 예외처리1
            estimated_wav = estimated_wav[..., 0:1]

        wav = self.reconst_wav(estimated_wav)

        return wav  # shape [B, samples]

    ### stft → wav ### !!
    def inverse_stft(self, stft_mag, stft_complex):  # ts[1,F,T], ts[1,F,T] → ts[1,N]
        stft_mag = stft_mag.squeeze(0)
        stft_complex = stft_complex.squeeze(0)
        assert stft_mag.shape == stft_complex.shape
        assert stft_mag.shape == (self.n_freq, self.n_times), f"{stft_mag.shape}"
        
        stft_mag = stft_mag.to(self.device)

        eps=1e-5
        phase = stft_complex / (stft_complex.abs() + eps)
        masked_stft_complex = stft_mag * phase

        estimated_wav = torch.istft(
                masked_stft_complex.to(self.device),
                n_fft=self.filter_length,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=torch.hann_window(1024).to(self.device),
                normalized=False,
                onesided=True
            )
        
        # pad_size 부분 제거
        # forward에서 reflect pad를 (pad_size, pad_size)만큼 했었으므로,
        # 최종 waveform에서 앞뒤로 pad_size samples씩 잘라낸다.
        pad_size = 432
        if estimated_wav.shape[-1] > pad_size*2:
            estimated_wav = estimated_wav[..., pad_size:-pad_size]  # shape [B, samples]
        else:
            # 혹시 길이가 매우 짧다면 예외처리
            estimated_wav = estimated_wav[..., 0:1]
        wav = self.reconst_wav(estimated_wav)
        return wav  # np[1,N]

def affusion_normalize_spectrogram(
    spectrogram: torch.Tensor,
    max_value: float = 200, 
    min_value: float = 1e-5, 
    power: float = 1., 
    inverse: bool = False
) -> torch.Tensor:
    
    # Rescale to 0-1
    max_value = np.log(max_value) # 5.298317366548036
    min_value = np.log(min_value) # -11.512925464970229

    assert spectrogram.max() <= max_value and spectrogram.min() >= min_value

    data = (spectrogram - min_value) / (max_value - min_value)

    # Invert
    if inverse:
        data = 1 - data

    # Apply the power curve
    data = torch.pow(data, power)  
    
    # 1D -> 3D
    data = data.repeat(3, 1, 1)

    # Flip Y axis: image origin at the top-left corner, spectrogram origin at the bottom-left corner
    data = torch.flip(data, [1])

    return data

def auffusion_denormalize_spectrogram(
    data: torch.Tensor,
    max_value: float = 200, 
    min_value: float = 1e-5, 
    power: float = 1, 
    inverse: bool = False,
) -> torch.Tensor:
    
    max_value = np.log(max_value)
    min_value = np.log(min_value)

    # Flip Y axis: image origin at the top-left corner, spectrogram origin at the bottom-left corner
    data = torch.flip(data, [1])

    assert len(data.shape) == 3, "Expected 3 dimensions, got {}".format(len(data.shape))
    
    if data.shape[0] == 1:
        data = data.repeat(3, 1, 1)
        
    assert data.shape[0] == 3, "Expected 3 channels, got {}".format(data.shape[0])
    data = data[0]

    # Reverse the power curve
    data = torch.pow(data, 1 / power)

    # Invert
    if inverse:
        data = 1 - data

    # Rescale to max value
    spectrogram = data * (max_value - min_value) + min_value

    return spectrogram
    
def normalize(images):
    """
    Normalize an image array to [-1,1].
    """
    if images.min() >= 0:
        return 2.0 * images - 1.0
    else:
        return images

def pad_spec(spec, spec_length, pad_value=0, random_crop=True): # spec: [3, mel_dim, spec_len]
    assert spec_length % 8 == 0, "spec_length must be divisible by 8"
    if spec.shape[-1] < spec_length:
        # pad spec to spec_length
        spec = F.pad(spec, (0, spec_length - spec.shape[-1]), value=pad_value)
    else:
        # random crop
        if random_crop:
            start = random.randint(0, spec.shape[-1] - spec_length)
            spec = spec[:, :, start:start+spec_length]
        else:
            spec = spec[:, :, :spec_length]
    return spec
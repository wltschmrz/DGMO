import torchaudio

def load_audio_torch(source_path, sampling_rate, mono=True):
    waveform, sr = torchaudio.load(source_path, normalize=True)  # librosa처럼 float32 [-1, 1]로 로드
    waveform = waveform.mean(dim=0) if (waveform.shape[0] > 1) and mono else waveform  # mono 변환
    waveform = torchaudio.functional.resample(waveform, sr, sampling_rate) if sr != sampling_rate else waveform
    return waveform.numpy().squeeze(0), sampling_rate

import os
import sys

proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(proj_dir, 'src_audioldm')
sys.path.extend([proj_dir, src_dir])

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import matplotlib.pyplot as plt
import soundfile as sf
import re
from models.mask import Mask
from utils import calculate_sisdr, calculate_sdr

def segment_wav(waveform, target_length, start=0):  # [1,N+] → [1,N]
    waveform_length = waveform.shape[-1]
    assert waveform_length > 100, f"Waveform is too short, {waveform_length}"
    # Too short
    if waveform_length <= target_length:
        return waveform
    segment = waveform[:, start:start + target_length]
    return segment

def pad_wav(waveform, target_length):  # [1,N-] → [1,N]
    waveform_length = waveform.shape[-1]
    assert waveform_length > 100, f"Waveform is too short, {waveform_length}"
    # if same length
    if waveform_length == target_length:
        return waveform
    # padding (target이 더 긴때만 처리하면 됨)
    padded_wav = torch.zeros((1, target_length))
    padded_wav[:, :waveform_length] = waveform
    return padded_wav

def read_wav_file(filename):  # fname → np[1,N]
    # 1. file load
    wav, raw_sr = torchaudio.load(filename, normalize=True)  # ts[C,N'±]
    # 2. to mono channel
    wav = wav.mean(dim=0) if wav.shape[0] > 1 else wav  # ts[1,N'±]
    # 2. segment & padding (to target length)
    raw_length = int(raw_sr * 10.24)
    wav = segment_wav(wav, raw_length)  # ts[1,N'-]
    wav = pad_wav(wav, raw_length)      # ts[1,N']
    # 3. resampling
    wav = torchaudio.functional.resample(wav, raw_sr, 48000)  # ts[1,N]
    return wav.numpy()  # np[1,N]

def inference(audioldm, processor, target_path, mixed_path, config, file_id=None):
    device = audioldm.device
    learning_rate = config['learning_rate']
    num_epochs = config['num_epochs']
    batchsize = config['batchsize']
    strength = config['strength']
    iteration = config['iteration']
    text = config['text']
    text = re.sub(r'[\/:*?"<>|]', ' ', text)

    steps = config['steps']
    mixed_text = ""
    
    iter_sisdrs = []
    iter_sdris = []
    
    mask = Mask(device, 1, 1025, 1024)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(mask.parameters(), lr=learning_rate)

    for iter in range(iteration):
        processor.duration = 10.24
        target_wav = processor.read_wav_file(target_path)
        mixed_wav = processor.read_wav_file(mixed_path)
        target_wav = np.concatenate([target_wav, target_wav], axis=1)
        mixed_wav = np.concatenate([mixed_wav, mixed_wav], axis=1)
        assert mixed_wav.ndim == 2 and mixed_wav.shape[1] == 327680, mixed_wav.shape

        mixed_wav_ = processor.prepare_wav(mixed_wav)
        mixed_stft, mixed_stft_c = processor.wav_to_stft(mixed_wav_)

        mixed_mel = processor.wav_to_mel(mixed_wav_)
        batch_split = 1
        batchsize_ = batchsize // batch_split
        mixed_mels = mixed_mel.repeat(batchsize_, 1, 1, 1)
        ref_mels = mixed_mels

        # mixed_audio_for_embeds = None
        mixed_audio_for_embeds=read_wav_file(mixed_path)

        if iter != 0:
            masked_wav = processor.read_wav_file(masked_path)
            masked_wav_ = processor.prepare_wav(masked_wav)
            masked_stft, masked_stft_c = processor.wav_to_stft(masked_wav_)
            masked_mel = processor.wav_to_mel(masked_wav_)
            batch_split = 1
            batchsize_ = batchsize // batch_split
            masked_mels = mixed_mel.repeat(batchsize_, 1, 1, 1)
            ref_mels = masked_mels

            # mixed_audio_for_embeds = None
            mixed_audio_for_embeds=read_wav_file(masked_path)

        if iter == 0:
            mel_sample_list=[]
            for i in range(batch_split):
                mel_samples = audioldm.edit_audio_with_ddim_inversion_sampling(
                            mel=ref_mels,
                            original_text=mixed_text,
                            text=text,
                            duration=10.24,
                            batch_size=batchsize_,
                            transfer_strength=strength,
                            guidance_scale=2.5,
                            ddim_steps=steps,
                            clipping = False,
                            return_type="mel",
                            mixed_audio_for_embeds=mixed_audio_for_embeds
                        )
                mel_sample_list.append(mel_samples)
            
        elif iter != 0:
            mel_sample_list=[]
            for i in range(batch_split):
                mel_samples = audioldm.edit_audio_with_ddim_inversion_sampling(
                            mel=ref_mels,
                            original_text=mixed_text,
                            text=text,
                            duration=10.24,
                            batch_size=batchsize_,
                            transfer_strength=strength,
                            guidance_scale=2.5,
                            ddim_steps=steps,
                            clipping = False,
                            return_type="mel",
                            mixed_audio_for_embeds=mixed_audio_for_embeds,
                        )
                mel_sample_list.append(mel_samples)
        mel_samples = torch.cat(mel_sample_list, dim=0)
        assert mel_samples.size(0) == batchsize and mel_samples.dim() == 4, (mel_samples.shape, batchsize)

        batch_sample = 0
        wav_sample = processor.inverse_mel_with_phase(mel_samples[batch_sample:batch_sample+1], mixed_stft_c)
        wav_sample = wav_sample.squeeze()
        sf.write(f'./test/batch_samples/edited_{text}_{file_id}_{strength:.4f}_{iter}_{batch_sample}.wav', wav_sample, 16000)

        # ------------------------------------------------------------------ #

        sisdrs_list = []
        sdris_list = []
        loss_values = []

        for epoch in range(num_epochs):

            optimizer.zero_grad()  # 그래디언트 초기화
            masked_stft = (mixed_stft - mixed_stft.min()) * mask() + mixed_stft.min()  #ts[1,513,1024]
            
            masked_mel = processor.masked_stft_to_masked_mel(masked_stft)  # [1,1,1024,512]
            masked_mel_expended = masked_mel.repeat(batchsize, 1, 1, 1)

            loss = criterion(mel_samples, masked_mel_expended)  # 손실 계산

            loss.backward()  # 역전파
            optimizer.step()  # 가중치 업데이트

            loss_values.append(loss.item())  # 손실값 저장

            wav_sep = processor.inverse_stft(masked_stft, mixed_stft_c)
            target_wav = target_wav.squeeze(0)
            mixed_wav = mixed_wav.squeeze(0)
            assert len(wav_sep) <= len(target_wav), (len(wav_sep), len(target_wav))
            wav_src = target_wav[:len(wav_sep)]
            wav_mix = mixed_wav[:len(wav_sep)]

            sdr_no_sep = calculate_sdr(ref=wav_src, est=wav_mix)
            sdr = calculate_sdr(ref=wav_src, est=wav_sep)
            sdri = sdr - sdr_no_sep
            sisdr = calculate_sisdr(ref=wav_src, est=wav_sep)

            sisdrs_list.append(sisdr)
            sdris_list.append(sdri)
            target_wav = np.expand_dims(target_wav, axis=0)
            mixed_wav = np.expand_dims(mixed_wav, axis=0)
                
        # ------------------------------------------------------------------ #

        plt.plot(loss_values)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Trend')
        plt.savefig(f'./test/plot/loss_{text}_{file_id}_{iter}.png')
        plt.close()

        plt.plot(sisdrs_list)
        plt.xlabel('Epoch')
        plt.ylabel('sisdr')
        plt.title('sisdr Trend')
        plt.savefig(f'./test/plot/sisdr_{text}_{file_id}_{iter}.png')
        plt.close()

        plt.plot(sdris_list)
        plt.xlabel('Epoch')
        plt.ylabel('sdri')
        plt.title('sdri Trend')
        plt.savefig(f'./test/plot/sdri_{text}_{file_id}_{iter}.png')
        plt.close()

        iter_sisdrs.append(sisdr)
        iter_sdris.append(sdri)

        wav_sep = processor.inverse_stft(masked_stft, mixed_stft_c)

        sf.write(f'./test/result/{text}_{file_id}_{iter}.wav', wav_sep, 16000)

        masked_path = f'./test/result/{text}_{file_id}_{iter}.wav'

    return iter_sisdrs, iter_sdris

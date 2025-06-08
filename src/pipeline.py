import os
import sys
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(proj_dir)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.models import AudioLDM
from src.models import Mask
from src.utils import load_config, save_wav_file, AudioDataProcessor, make_unique_dir
import argparse

class DGMO(nn.Module):
    def __init__(self, *, config_path="./configs/DGMO.yaml", device="cuda", **kwargs):
        super(DGMO, self).__init__()
        self.device = torch.device(device)

        config = load_config(config_path) if config_path else {}
        for key in list(kwargs):
            if key in config:
                config[key] = kwargs.pop(key)
        self._apply_config(config)
        
        ldm_config = load_config(self.ldm_config_path) if self.ldm_config_path else {}
        for key, value in ldm_config.items():
            if key == "repo_id":
                setattr(self, key, value)

        repo_type = self.get_model_type(self.repo_id)
        match repo_type:
            case "audioldm":
                self.ldm = AudioLDM(ckpt=self.repo_id, device=self.device, **kwargs)

            case _:
                raise ValueError(f"Invalid repo_id: {self.repo_id}")
            
        self.processor = AudioDataProcessor(config_path=self.ldm_config_path, device=self.device, **kwargs)

        self.ldm.eval()  ##
        for param in self.ldm.parameters():
            param.requires_grad = False 


    def _apply_config(self, config):
        for key, value in config.items():
            if isinstance(value, dict):
                self._apply_config(value)
            else:
                setattr(self, key, value)
    
    def get_model_type(self, repo_id):
        "Assume that repo_id be 'file/model_name-repo_name'"
        base_name = repo_id.rsplit("/", 1)[-1]
        model_name = base_name.split("-")[0]
        return model_name
    
    def inference(
            self, 
            mix_wav_path: str = None, 
            text: str = None, 
            save_dir="./test",
            save_fname="sample.wav",
            thresholding=True,
            ):
        assert isinstance(text, str), "text must be a str"
        os.makedirs(save_dir, exist_ok=True)

        mask = Mask(channel=1,
            height=self.processor.n_freq_bins,
            width=self.processor.n_time_frames,
            device=self.device)
        optimizer = optim.Adam(mask.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        assert self.ddim_batch % self.num_splits == 0,\
            "ddim_batch must be divisible by batch_split"
        chunks = self.ddim_batch // self.num_splits  # 몫

        mix_wav = self.processor.read_wav_file(mix_wav_path)
        assert mix_wav.shape[1] == self.processor.sample_length, mix_wav.shape
        mix_wav_norm = self.processor.prepare_wav(mix_wav)
        mix_stft_mag, mix_stft_complex = self.processor.wav_to_stft(mix_wav_norm)
        cur_stft_mag = mix_stft_mag

        for iter in range(self.iteration):
            cur_mel = self.processor.stft_to_mel(cur_stft_mag)
            vae_input = self.processor.preprocess_spec(cur_mel)
            vae_inputs = vae_input.repeat(chunks, 1, 1, 1)

            # ----- Reference Generation ----- #
            mel_sample_list=[]
            for _ in range(self.num_splits):
                ref_mels = self.ldm.edit(
                    mel=vae_inputs,
                    inv_text=[""],
                    text=[text],
                    ddim_steps=self.ddim_steps,
                    timestep_level=self.noise_level,
                    guidance_scale=self.guidance_scale,
                    batch_size=chunks,
                    duration=10.24,
                )
                
                mel_sample_list.append(ref_mels)

            ref_mels = torch.cat(mel_sample_list, dim=0)
            assert ref_mels.size(0) == self.ddim_batch and ref_mels.dim() == 4,\
                (ref_mels.shape, self.ddim_batch)

            # ------- Mask Optimization ------- #
            loss_values = []
            for epoch in range(self.epochs):
                optimizer.zero_grad()
                masked_stft = mix_stft_mag * mask()  #ts[1,513,1024]
                masked_mel = self.processor.stft_to_mel(masked_stft)  # [1,M,T]
                msked_vae_mel = self.processor.preprocess_spec(masked_mel)  # [1,1,T*,M*]
                msked_vae_mels = msked_vae_mel.repeat(self.ddim_batch, 1, 1, 1)

                loss = criterion(ref_mels, msked_vae_mels)
                loss.backward()
                optimizer.step()
                loss_values.append(loss.item())
            
            with torch.no_grad():
                final_mask = mask().detach().clone()
                if thresholding:
                    threshold = 0.9
                    final_mask[final_mask >= threshold] = 1.0
                cur_stft_mag = mix_stft_mag * final_mask  #ts[1,513,1024]
                
        msked_wav = self.processor.inverse_stft(cur_stft_mag, mix_stft_complex)

        save_path = os.path.join(save_dir, save_fname)
        save_wav_file(filename=save_path, wav_np=msked_wav, target_sr=self.processor.sampling_rate)
        
        return msked_wav, ref_mels.mean(dim=0)  # np[1,N]  ####


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default="./configs/DGMO.yaml")
    parser.add_argument('--mix_wav_path', type=str, required=True)
    parser.add_argument('--text', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default="./results")
    parser.add_argument('--save_fname', type=str, default="sample.wav")
    parser.add_argument('--device', type=str, default="cuda:0")
    args = parser.parse_args()

    dgmo = DGMO(config_path=args.config_path, device=args.device)

    dgmo.inference(
        mix_wav_path=args.mix_wav_path,
        text=args.text,
        save_dir=args.save_dir,
        save_fname=args.save_fname
    )
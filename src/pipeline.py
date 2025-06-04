import os
import sys
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(proj_dir, 'src')

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.models import AudioLDM
from src.models import Mask
from src.utils import load_config, save_wav_file, AudioDataProcessor, make_unique_dir

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
            # case "audioldm2":
            #     self.ldm = AudioLDM2(ckpt=self.repo_id, device=self.device, **kwargs)
            # case "auffusion":
            #     self.ldm = Auffusion(ckpt=self.repo_id, device=self.device, **kwargs)
            case _:
                raise ValueError(f"Invalid repo_id: {self.repo_id}")
            
        self.processor = AudioDataProcessor(config_path=self.ldm_config_path, device=self.device, **kwargs)

        self.ldm.eval()  ##
        for param in self.ldm.parameters():
            param.requires_grad = False 

        # for param in self.ldm.vae.parameters():
        #     param.requires_grad = False  
        # for param in self.ldm.stft.parameters():
        #     param.requires_grad = False 
        # for param in self.ldm.model.parameters():
        #     param.requires_grad = False  

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

        # ref_wav = self.ldm.vae.decode_to_waveform(ref_mels)  ####
        # print(ref_wav.shape)
        # ref_wav = ref_wav.mean(axis=0)
        # print(ref_wav.shape)
        # # try:
        # ref_wav = np.expand_dims(ref_wav, axis=0)
        # # except: pass
        # print(ref_wav.shape)
        # # ref_wav = self.processor.prepare_wav(ref_wav)
        # # print(ref_wav.shape)
        # ref_wav = ref_wav.astype(np.float32)  # float64 → float32
        # ref_wav = ref_wav / 32768.0
        # save_path = os.path.join(save_dir, save_fname)
        # save_path_ref = os.path.join(save_dir, "ref.wav")
        # save_wav_file(filename=save_path, wav_np=msked_wav, target_sr=self.processor.sampling_rate)
        # save_wav_file(filename=save_path_ref, wav_np=ref_wav, target_sr=self.processor.sampling_rate)
        
        return msked_wav, ref_mels.mean(dim=0)  # np[1,N]  ####

if __name__ == "__main__":

    config = "./configs/DGMO.yaml"

    mix_path = "./data/samples/Cat_n_Footstep.wav"
    ref_paths = [
        "./data/samples/A_cat_meowing.wav",
        "./data/samples/Foot_steps_on_the_wooden_floor.wav"
    ]
    texts = [
        "A cat meowing",
        "Foot steps on the wooden floor"
    ]

    dgmo = DGMO(config_path=config, device="cuda",
                # iteration=iter
                )


    ########## TESTING BASIC DGMO ##########
    
    save_dir = make_unique_dir("./test/results", "single")
    mel_paths = [
        os.path.join(save_dir, f"cat.png"),
        os.path.join(save_dir, f"step.png")
    ]
    sep_paths = [
        os.path.join(save_dir, f"cat.wav"),
        os.path.join(save_dir, f"step.wav")
    ]

    for i in range(2):
        sep_wav = dgmo.inference(
            mix_wav_path=mix_path,
            text=texts[i],
            save_path=sep_paths[i]
        )

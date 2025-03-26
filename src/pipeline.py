# mixed audio 파일 하나 가져와서 경로랑 텍스트 주면 결과값 주는거
# result/ 에 뭐 줄거임?? >> mixed audio랑 sep의 mel spec이랑, ref, sep의 wav

import os
import sys
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(proj_dir, 'src')
sys.path.extend([proj_dir, src_dir])
import torch
import torch.nn as nn
import torch.optim as optim
from src.models import AudioLDM, AudioLDM2, Auffusion, Mask, Multi_Class_Mask
from src.utils import load_config, save_wav_file, AudioDataProcessor

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
            case "audioldm2":
                self.ldm = AudioLDM2(ckpt=self.repo_id, device=self.device, **kwargs)
            case "auffusion":
                self.ldm = Auffusion(ckpt=self.repo_id, device=self.device, **kwargs)
            case _:
                raise ValueError(f"Invalid repo_id: {self.repo_id}")
            
        self.processor = AudioDataProcessor(config_path=self.ldm_config_path, device=self.device, **kwargs)

        self.ldm.eval()
        for param in self.ldm.parameters():
            param.requires_grad = False  # 모든 가중치가 학습되지 않음

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
            save_path="./test/sample.wav",
            thresholding=True,
            ):
        assert isinstance(text, str), "text must be a str"

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
                ref_mels = self.ldm.ddim_inv_editing(
                    mel=vae_inputs,
                    original_text="",
                    text=text,
                    duration=self.processor.duration,
                    batch_size=chunks,
                    timestep_level=self.noise_level,
                    guidance_scale=self.guidance_scale,
                    ddim_steps=self.ddim_steps,
                    return_type="mel",  # "ts"/"np"/"mel"
                    mel_clipping = False,
                )
                mel_sample_list.append(ref_mels)

            ref_mels = torch.cat(mel_sample_list, dim=0)
            assert ref_mels.size(0) == self.ddim_batch and ref_mels.dim() == 4,\
                (ref_mels.shape, self.ddim_batch)

            # ------- Mask Optimization ------- #
            loss_values = []
            for epoch in range(self.epochs):
                optimizer.zero_grad()
                masked_stft = (mix_stft_mag - mix_stft_mag.min()) * mask() + mix_stft_mag.min()  #ts[1,513,1024]
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
                    threshold = 0.8
                    final_mask[final_mask >= threshold] = 1.0
                cur_stft_mag = (mix_stft_mag - mix_stft_mag.min()) * final_mask + mix_stft_mag.min()  #ts[1,513,1024]
                
        msked_wav = self.processor.inverse_stft(cur_stft_mag, mix_stft_complex)
        save_wav_file(filename=save_path, wav_np=msked_wav, target_sr=self.processor.sampling_rate)
        return msked_wav  # np[1,N]

    def joint_opt_inference(
            self, 
            mix_wav_path: str = None, 
            text: list = None, 
            save_dir="./test/mixed_id"
            ):
        assert isinstance(text, list) and len(text) >= 2,\
            "text must be a list with at least 2 elements"
        os.makedirs(save_dir, exist_ok=True)
        
        num_class = len(text)
        mask = Multi_Class_Mask(
            num_classes=num_class,
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
        cur_all_stfts = [mix_stft_mag for _ in range(num_class)]

        for iter_idx in range(self.iteration):
            all_ref_mels = []  # list of [ddim_batch, 1, T*, M*]
            for cls_idx, class_text in enumerate(text):
                cur_stft_mag = cur_all_stfts[cls_idx]  # np[1, N]
                cur_mel = self.processor.stft_to_mel(cur_stft_mag)
                vae_input = self.processor.preprocess_spec(cur_mel)
                vae_inputs = vae_input.repeat(chunks, 1, 1, 1)  # [chunk, 1, T*, M*]
            
            # ----- Reference Generation ----- #
                mel_sample_list=[]
                for _ in range(self.num_splits):
                    ref_mels = self.ldm.ddim_inv_editing(
                        mel=vae_inputs,
                        original_text="",
                        text=class_text,
                        duration=self.processor.duration,
                        batch_size=chunks,
                        timestep_level=self.noise_level,
                        guidance_scale=self.guidance_scale,
                        ddim_steps=self.ddim_steps,
                        return_type="mel",  # "ts"/"np"/"mel"
                        mel_clipping = False,
                    )
                    mel_sample_list.append(ref_mels)  # [chunk, 1, T*, M*]
                
                single_ref_mels = torch.cat(mel_sample_list, dim=0)  # [batch, 1, T*, M*]
                assert single_ref_mels.size(0) == self.ddim_batch and single_ref_mels.dim() == 4,\
                    (single_ref_mels.shape, self.ddim_batch)
                
                all_ref_mels.append(single_ref_mels)  # [len(text) * [batch, 1, T*, M*]]
            
            # ------- Mask Optimization ------- #
            loss_values = []
            for epoch in range(self.epochs):
                optimizer.zero_grad()
                masking = mask()  # [num_class, H, W]
                total_loss = 0.0
                for cls_idx in range(num_class):
                    single_mask = masking[cls_idx:cls_idx+1, :, :]  # [1, H, W]
                    
                    single_msked_stft = (mix_stft_mag - mix_stft_mag.min()) * single_mask + mix_stft_mag.min()
                    single_msked_mel = self.processor.stft_to_mel(single_msked_stft)
                    single_msked_vae_mel = self.processor.preprocess_spec(single_msked_mel)
                    
                    single_msked_vae_mels = single_msked_vae_mel.repeat(self.ddim_batch, 1, 1, 1)  # [batch, 1, T*, M*]
                    single_ref_mels = all_ref_mels[cls_idx]
                    
                    cls_loss = criterion(single_ref_mels, single_msked_vae_mels)
                    total_loss += cls_loss

                total_loss.backward()
                optimizer.step()
                loss_values.append(total_loss.item())

            with torch.no_grad():
                final_mask = mask().detach().clone()  # [num_class, H, W]
                
                new_msked_stfts = []
                for cls_idx in range(num_class):
                    single_msked_stft = (mix_stft_mag - mix_stft_mag.min()) * final_mask[cls_idx:cls_idx+1] + mix_stft_mag.min()  #ts[1,513,1024]
                    new_msked_stfts.append(single_msked_stft)
                cur_all_stfts = new_msked_stfts

        est_wavs = []
        for cls_idx, single_msked_stft in enumerate(new_msked_stfts):
            rm = True if cls_idx+1 == len(new_msked_stfts) else False
            est_wav = self.processor.inverse_stft(single_msked_stft, mix_stft_complex, fac_rm=rm)
            est_wavs.append(est_wav)
            save_path = os.path.join(save_dir, f"sep_{cls_idx}.wav")
            save_wav_file(filename=save_path,
                wav_np=est_wav,
                target_sr=self.processor.sampling_rate)
        return est_wavs    # [class * np[1,N]]

if __name__ == "__main__":
    from src.utils import read_wav_file, plot_wav_mel, printing_sdrs, make_unique_dir

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

    dgmo = DGMO(config_path=config, device="cuda:1",
                # iteration=iter
                )

    ########## TESTING BASIC DGMO ##########
    '''
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

        mix_wav = read_wav_file(filename=mix_path, target_duration=10.24, target_sr=16000)
        ref_wav = read_wav_file(filename=ref_paths[i], target_duration=10.24, target_sr=16000)

        scores = printing_sdrs(ref=ref_wav, mix=mix_wav, est=sep_wav)
        wav_paths = [mix_wav, sep_wav, ref_wav]
        plot_wav_mel(wav_paths, save_path=mel_paths[i], score=scores, config_path=config)
    '''
    ########## TESTING JOINTLY OPT ##########

    # for iter in range(1, 11):
    save_dir = make_unique_dir("./test/results", "jointly")

    sep_wavs = dgmo.joint_opt_inference(
        mix_wav_path=mix_path,
        text=texts,
        save_dir=save_dir,
    )

    sep_path1 = os.path.join(save_dir, f"sep_0.wav")
    sep_path2 = os.path.join(save_dir, f"sep_1.wav")

    mix_wav = read_wav_file(filename=mix_path, target_duration=10.24, target_sr=16000)
    ref_wav1 = read_wav_file(filename=ref_paths[0], target_duration=10.24, target_sr=16000)
    ref_wav2 = read_wav_file(filename=ref_paths[1], target_duration=10.24, target_sr=16000)
    sep_wav1, sep_wav2 = sep_wavs

    scores = printing_sdrs(ref=ref_wav1, mix=mix_wav, est=sep_wav1)
    wav_paths = [mix_wav, sep_wav1, ref_wav1]
    png_save_path = os.path.join(save_dir, f"sep_0.png")
    plot_wav_mel(wav_paths, save_path=png_save_path, score=scores, config_path=config,
                # iteration=iter
                )
    scores = printing_sdrs(ref=ref_wav2, mix=mix_wav, est=sep_wav2)
    wav_paths = [mix_wav, sep_wav2, ref_wav2]
    png_save_path = os.path.join(save_dir, f"sep_1.png")
    plot_wav_mel(wav_paths, save_path=png_save_path, score=scores, config_path=config,
                # iteration=iter
                )

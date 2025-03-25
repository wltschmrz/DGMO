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
                self.channel = 1
            case "audioldm2":
                self.ldm = AudioLDM2(ckpt=self.repo_id, device=self.device, **kwargs)
                self.channel = 1
            case "auffusion":
                self.ldm = Auffusion(ckpt=self.repo_id, device=self.device, **kwargs)
                self.channel = 3
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
    
    def inference(self, mix_wav_path=None, text=None, save_path="./test/sample.wav"):
        mask = Mask(
            channel=1,
            height=self.processor.n_freq_bins,
            width=self.processor.n_time_frames,
            device=self.device
            )
        optimizer = optim.Adam(mask.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        for iter in range(self.iteration):
            if iter == 0:
                assert self.ddim_batchsize % self.batch_split == 0, "ddim_batchsize must be divisible by batch_split"
                batch = self.ddim_batchsize // self.batch_split

                mix_wav = self.processor.read_wav_file(mix_wav_path)
                msked_wav = None
                assert mix_wav.ndim == 2 and mix_wav.shape[1] == self.processor.sample_length, mix_wav.shape

                mix_wav_norm = self.processor.prepare_wav(mix_wav)
                mix_stft_mag, mix_stft_complex = self.processor.wav_to_stft(mix_wav_norm)
                mix_mel = self.processor.stft_to_mel(mix_stft_mag)
                vae_input = self.processor.preprocess_spec(mix_mel)
                vae_inputs = vae_input.repeat(batch, 1, 1, 1)
            
            elif iter != 0:
                assert msked_wav is not None, "msked_wav must be provided for iter > 0"
                msked_wav_norm = self.processor.prepare_wav(msked_wav)
                msked_stft_mag, msked_stft_complex = self.processor.wav_to_stft(msked_wav_norm)
                msked_mel = self.processor.stft_to_mel(msked_stft_mag)
                vae_input = self.processor.preprocess_spec(msked_mel)
                vae_inputs = vae_input.repeat(batch, 1, 1, 1)

            mel_sample_list=[]
            for i in range(self.batch_split):
                if "audioldm" in self.repo_id:
                    ref_mels = self.ldm.ddim_inv_editing(
                        mel=vae_inputs,
                        original_text="",
                        text=text,
                        duration=self.processor.duration,
                        batch_size=batch,
                        timestep_level=self.noise_level,
                        guidance_scale=self.guidance_scale,
                        ddim_steps=self.ddim_steps,
                        return_type="mel",  # "ts"/"np"/"mel"
                        mel_clipping = False,
                    )
                elif "auffusion" in self.repo_id:    #### 1. 이거 통합하는거랑, 2. 마스크 채널이랑, 3. 차원 수 안 맞는거 확인
                    print("auffusion")##
                    print(vae_inputs.shape)##
                    ref_mels = self.ldm.edit_audio_with_ddim_inversion_sampling(
                        mel=vae_inputs,
                        original_text="",
                        text=text,
                        duration=10.24,
                        batch_size=batch,
                        transfer_strength=self.noise_level,
                        guidance_scale=self.guidance_scale,
                        ddim_steps=self.ddim_steps,
                        return_type="mel",  # "ts"/"np"/"mel"
                        clipping=False,
                    )
                # print(ref_mels.shape)##
                mel_sample_list.append(ref_mels)

            ref_mels = torch.cat(mel_sample_list, dim=0)
            assert ref_mels.size(0) == self.ddim_batchsize and ref_mels.dim() == 4, (ref_mels.shape, self.ddim_batchsize)
            # print(ref_mels.shape)##
            # ------------------------------------------------------------------ #

            loss_values = []

            for epoch in range(self.epochs):
                optimizer.zero_grad()
                masked_stft = (mix_stft_mag - mix_stft_mag.min()) * mask() + mix_stft_mag.min()  #ts[1,513,1024]
                assert masked_stft.size() == (1, self.processor.n_freq_bins, self.processor.n_time_frames), masked_stft.size()
                masked_mel = self.processor.stft_to_mel(masked_stft)  # [1,M,T]
                masked_mel = self.processor.preprocess_spec(masked_mel)  # [1,1,T*,M*]
                masked_mel_expended = masked_mel.repeat(self.ddim_batchsize, 1, 1, 1)

                loss = criterion(ref_mels, masked_mel_expended)

                loss.backward()
                optimizer.step()

                loss_values.append(loss.item())
            
            with torch.no_grad():
                threshold = 0.8
                final_mask = mask().detach().clone()
                final_mask[final_mask >= threshold] = 1.0

                masked_stft = (mix_stft_mag - mix_stft_mag.min()) * final_mask + mix_stft_mag.min()  #ts[1,513,1024]
                msked_wav = self.processor.inverse_stft(masked_stft, mix_stft_complex)

        save_wav_file(filename=save_path, wav_np=msked_wav, target_sr=self.processor.sampling_rate)
        return msked_wav  # np[1,N]

    def joint_opt_inference(self, mix_wav_path=None, text=None, save_dir="./test/mixed_id"):
        """
        text: 각 소스(클래스)에 대한 prompt(문장) 리스트. 최소 2개 이상.
        save_dir: 최종 분리된 wav들을 저장할 폴더 경로(없으면 생성하도록 수정 가능).
        """
        os.makedirs(save_dir, exist_ok=True)
        assert isinstance(text, list) and len(text) >= 2, "text must be a list with at least 2 elements"
        num_class = len(text)
        
        mask = Multi_Class_Mask(
            num_classes=num_class,
            height=self.processor.n_freq_bins,
            width=self.processor.n_time_frames,
            device=self.device
            )
        optimizer = optim.Adam(mask.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        assert self.ddim_batchsize % self.batch_split == 0, "ddim_batchsize must be divisible by batch_split"
        batch = self.ddim_batchsize // self.batch_split
        
        mix_wav = self.processor.read_wav_file(mix_wav_path)
        assert mix_wav.ndim == 2 and mix_wav.shape[1] == self.processor.sample_length, mix_wav.shape
        mix_wav_norm = self.processor.prepare_wav(mix_wav)
        mix_stft_mag, mix_stft_complex = self.processor.wav_to_stft(mix_wav_norm)
        prev_iter_class_stfts = [mix_stft_mag for _ in range(num_class)]

        for iter_idx in range(self.iteration):

            ref_mels_for_all_classes = []  # list of [ddim_batchsize, 1, T*, M*]
            for class_idx, class_text in enumerate(text):
                cur_class_stft_mag = prev_iter_class_stfts[class_idx]  # np[1, N]
                cur_mel = self.processor.stft_to_mel(cur_class_stft_mag)
                vae_input = self.processor.preprocess_spec(cur_mel)
                vae_inputs = vae_input.repeat(batch, 1, 1, 1)  # [batch, 1, T*, M*]
          
                mel_sample_list=[]
                for _ in range(self.batch_split):
                    if "audioldm" in self.repo_id:
                        ref_mels = self.ldm.ddim_inv_editing(
                            mel=vae_inputs,
                            original_text="",
                            text=class_text,
                            duration=self.processor.duration,
                            batch_size=batch,
                            timestep_level=self.noise_level,
                            guidance_scale=self.guidance_scale,
                            ddim_steps=self.ddim_steps,
                            return_type="mel",  # "ts"/"np"/"mel"
                            mel_clipping = False,
                        )
                    elif "auffusion" in self.repo_id:    #### 1. 이거 통합하는거랑, 2. 마스크 채널이랑, 3. 차원 수 안 맞는거 확인
                        ref_mels = self.ldm.edit_audio_with_ddim_inversion_sampling(
                            mel=vae_inputs,
                            original_text="",
                            text=class_text,
                            duration=self.processor.duration,
                            batch_size=batch,
                            transfer_strength=self.noise_level,
                            guidance_scale=self.guidance_scale,
                            ddim_steps=self.ddim_steps,
                            return_type="mel",  # "ts"/"np"/"mel"
                            clipping=False,
                        )
                    mel_sample_list.append(ref_mels)  # [freg, 1, T*, M*]
                
                ref_mels_for_class = torch.cat(mel_sample_list, dim=0)  # [batch, 1, T*, M*]
                assert ref_mels_for_class.size(0) == self.ddim_batchsize and ref_mels_for_class.dim() == 4,\
                    (ref_mels_for_class.shape, self.ddim_batchsize)
                
                ref_mels_for_all_classes.append(ref_mels_for_class)  # [len(text) * [batch, 1, T*, M*]]
            # ------------------------------------------------------------------ #

            loss_values = []

            for epoch in range(self.epochs):
                optimizer.zero_grad()
                mask_out = mask()  # [num_class, H, W]
                total_loss = 0.0
                for class_idx in range(num_class):
                    mask_for_class = mask_out[class_idx:class_idx+1, :, :]  # [1, H, W]
                    
                    masked_stft_for_class = (mix_stft_mag - mix_stft_mag.min()) * mask_for_class + mix_stft_mag.min()
                    assert masked_stft_for_class.size() == (1, self.processor.n_freq_bins, self.processor.n_time_frames),\
                        masked_stft_for_class.size()
                    
                    masked_mel_for_class = self.processor.stft_to_mel(masked_stft_for_class)
                    masked_mel_for_class = self.processor.preprocess_spec(masked_mel_for_class)
                    
                    masked_mel_for_class_expanded = masked_mel_for_class.repeat(self.ddim_batchsize, 1, 1, 1)  # [batch, 1, T*, M*]
                    
                    ref_mels_for_class = ref_mels_for_all_classes[class_idx]
                    class_loss = criterion(ref_mels_for_class, masked_mel_for_class_expanded)
                    total_loss += class_loss

                total_loss.backward()
                optimizer.step()
                loss_values.append(total_loss.item())

            with torch.no_grad():
                final_mask = mask().detach().clone()  # [num_class, H, W]
                
                new_msked_stfts = []
                for class_idx in range(num_class):
                    masked_stft_for_class = (mix_stft_mag - mix_stft_mag.min()) * final_mask[class_idx:class_idx+1] \
                        + mix_stft_mag.min()  #ts[1,513,1024]
                    new_msked_stfts.append(masked_stft_for_class)
                prev_iter_class_stfts = new_msked_stfts

        msked_wavs = []
        for class_idx, masked_stft_for_class in enumerate(new_msked_stfts):
            rm = True if class_idx+1 == len(new_msked_stfts) else False
            seped_wav = self.processor.inverse_stft(masked_stft_for_class, mix_stft_complex, fac_rm=rm)
            msked_wavs.append(seped_wav)
            save_path = os.path.join(save_dir, f"sep_{class_idx}.wav")
            save_wav_file(filename=save_path,
                wav_np=seped_wav,
                target_sr=self.processor.sampling_rate)
        return msked_wavs    # [class * np[1,N]]

if __name__ == "__main__":
    from src.utils import read_wav_file, plot_wav_mel, printing_sdrs
    
    mix = "./data/samples/Cat_n_Footstep.wav"
    # sep = "./test/result/cat_separated.wav"
    gt = "./data/samples/A_cat_meowing.wav"
    gt2 = "./data/samples/Foot_steps_on_the_wooden_floor.wav"

    text = "A cat meowing"

    config = "./configs/DGMO.yaml"

    for iter in range(1, 11):
        save_dir = f"./test/mel_test/jointly{iter}"
        sep_dir = f"./test/result/jointly{iter}"
        os.makedirs(save_dir, exist_ok=True)

        dgmo = DGMO(config_path=config, device="cuda:1", iteration=iter)
        # dgmo.inference(
        #     mix_wav_path=mix,
        #     text=text,
        #     save_path=sep
        # )
        text_li = [
            "A cat meowing",
            "Foot steps on the wooden floor"
        ]
        dgmo.joint_opt_inference(
            mix_wav_path=mix,
            text=text_li,
            save_dir=sep_dir,
        )

        sep_path1 = os.path.join(sep_dir, f"sep_0.wav")
        sep_path2 = os.path.join(sep_dir, f"sep_1.wav")

        mix_wav = read_wav_file(filename=mix, target_duration=10.24, target_sr=16000)
        # sep_wav = read_wav_file(filename=sep, target_duration=10.24, target_sr=16000)
        sep_wav1 = read_wav_file(filename=sep_path1, target_duration=10.24, target_sr=16000)
        sep_wav2 = read_wav_file(filename=sep_path2, target_duration=10.24, target_sr=16000)
        gt_wav = read_wav_file(filename=gt, target_duration=10.24, target_sr=16000)
        gt2_wav = read_wav_file(filename=gt2, target_duration=10.24, target_sr=16000)

        scores = printing_sdrs(ref=gt_wav, mix=mix_wav, est=sep_wav1)
        wav_paths = [mix_wav, sep_wav1, gt_wav]
        png_save_path = os.path.join(save_dir, f"sep_1.png")
        plot_wav_mel(wav_paths, save_path=png_save_path, score=scores, config_path=config, iteration=iter)

        scores = printing_sdrs(ref=gt2_wav, mix=mix_wav, est=sep_wav2)
        wav_paths = [mix_wav, sep_wav2, gt2_wav]
        png_save_path = os.path.join(save_dir, f"sep_2.png")
        plot_wav_mel(wav_paths, save_path=png_save_path, score=scores, config_path=config, iteration=iter)

        # scores = printing_sdrs(ref=gt_wav, mix=mix_wav, est=sep_wav)
        # wav_paths = [mix_wav, sep_wav, gt_wav]
        # png_save_path = f"./test/mel_test/cat--2.png"
        # plot_wav_mel(wav_paths, save_path=png_save_path, score=scores, config_path=config)
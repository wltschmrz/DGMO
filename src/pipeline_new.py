# mixed audio 파일 하나 가져와서 경로랑 텍스트 주면 결과값 주는거
# result/ 에 뭐 줄거임?? >> mixed audio랑 sep의 mel spec이랑, ref, sep의 wav

import os
import sys
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(proj_dir, 'src_audioldm')
sys.path.extend([proj_dir, src_dir])
import torch
import torch.nn as nn
import torch.optim as optim
from src.models import AudioLDM, AudioLDM2, Auffusion, Mask
from src.data_processing.audio_processing import save_wav_file, AudioDataProcessor
from src.utils import load_config

class DGMO(nn.Module):
    def __init__(self, *, config_path="./configs/DGMO.yaml", device="cuda", **kwargs):
        super(DGMO, self).__init__()
        self.device = torch.device(device)

        config = load_config(config_path) if config_path else {}
        config.update(kwargs)
        self._apply_config(config)
        
        ldm_config = load_config(self.ldm_config_path) if self.ldm_config_path else {}
        for key, value in ldm_config.items():
            if key == "repo_id":
                setattr(self, key, value)

        repo_type = self.get_model_type(self.repo_id)
        match repo_type:
            case "audioldm":
                self.ldm = AudioLDM(ckpt=self.repo_id, device=self.device)
                self.channel = 1
            case "audioldm2":
                self.ldm = AudioLDM2(ckpt=self.repo_id, device=self.device)
                self.channel = 1
            case "auffusion":
                self.ldm = Auffusion(ckpt=self.repo_id, device=self.device)
                self.channel = 3
            case _:
                raise ValueError(f"Invalid repo_id: {self.repo_id}")
            
        self.processor = AudioDataProcessor(config_path=self.ldm_config_path, device=self.device)

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

    def init_mask(self, channel=1, height=1024, width=513):
        return Mask(
            channel=channel,
            height=height,
            width=width,
            device=self.device
            )
    
    def inference(self, mix_wav_path=None, text=None, save_path="./test/sample.wav"):
        mask = self.init_mask(
            channel=self.channel,
            height=self.processor.n_freq,
            width=self.processor.target_length
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
                mel_sample_list.append(ref_mels)

            ref_mels = torch.cat(mel_sample_list, dim=0)
            assert ref_mels.size(0) == self.ddim_batchsize and ref_mels.dim() == 4, (ref_mels.shape, self.ddim_batchsize)

            # ------------------------------------------------------------------ #

            loss_values = []

            for epoch in range(self.epochs):
                optimizer.zero_grad()
                masked_stft = (mix_stft_mag - mix_stft_mag.min()) * mask() + mix_stft_mag.min()  #ts[1,513,1024]
                assert masked_stft.size() == (1, self.processor.n_freq,
                                              self.processor.target_length), masked_stft.size()
                masked_mel = self.processor.stft_to_mel(masked_stft)  # [1,M,T]
                masked_mel = self.processor.preprocess_spec(masked_mel)  # [1,1,T*,M*]
                masked_mel_expended = masked_mel.repeat(self.ddim_batchsize, 1, 1, 1)

                loss = criterion(ref_mels, masked_mel_expended)  # 손실 계산

                loss.backward()  # 역전파
                optimizer.step()  # 가중치 업데이트

                loss_values.append(loss.item())  # 손실값 저장
            
            masked_stft = (mix_stft_mag - mix_stft_mag.min()) * mask() + mix_stft_mag.min()  #ts[1,513,1024]
            msked_wav = self.processor.inverse_stft(masked_stft, mix_stft_complex)

        save_wav_file(filename=save_path, wav_np=msked_wav, target_sr=self.processor.sampling_rate)

if __name__ == "__main__":
    dgmo = DGMO(config_path="./configs/DGMO.yaml", device="cuda:1")
    dgmo.inference(
        mix_wav_path="./Cat_n_Footstep.wav",
        text="A cat meowing",
        save_path="./cat_separated.wav"
    )

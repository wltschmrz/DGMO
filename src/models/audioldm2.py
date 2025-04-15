from typing import Any, Callable, Dict, List, Optional, Union
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import repeat
from diffusers import (
    AudioLDM2Pipeline
)

# Suppress partial model loading warning
os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")

class AudioLDM2(nn.Module):
    
    def __init__(self, device='cuda', repo_id="cvssp/audioldm2-large", config=None):
        super().__init__()
        self.device = torch.device(device)
        pipe = AudioLDM2Pipeline.from_pretrained(repo_id, use_safetensors=False)

        # Setup components and move to device
        self.pipe = pipe.to(self.device)

        self.vae = self.pipe.vae
        self.scheduler = self.pipe.scheduler
        self.vocoder = self.pipe.vocoder
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder
        self.unet = self.pipe.unet


        self.evalmode = True
        self.checkpoint_path = repo_id
        self.audio_duration = 10.24 if not config else config['duration']
        self.original_waveform_length = int(self.audio_duration * self.vocoder.config.sampling_rate)  # 10.24 * 16000 = 163840
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)  # 4
        print(f'[INFO] audioldm.py: loaded AudioLDM!')

    def eval_(self):
        self.evalmode = True

    def train_(self):
        self.evalmode = False

    def encode_audios(self, x):  # ts[B, 1, T:1024, M:64] -> ts[B, C:8, lT:256, lM:16]
        encoder_posterior = self.vae.encode(x)
        unscaled_z = encoder_posterior.latent_dist.sample()
        z = unscaled_z * self.vae.config.scaling_factor  # Normalize z to have std=1 / factor: 0.9227914214134216
        return z

    def decode_latents(self, latents):  # ts[B, C:8, lT:256, lM:16] -> ts[B, 1, T:1024, M:64]
        latents = 1 / self.vae.config.scaling_factor * latents
        mel_spectrogram = self.vae.decode(latents).sample
        return mel_spectrogram

    def mel_to_waveform(self, mel_spectrogram):  # ts[B, 1, T:1024, M:64] -> ts[B, N:163872]
        if mel_spectrogram.dim() == 4:
            mel_spectrogram = mel_spectrogram.squeeze(1)
        elif mel_spectrogram.dim() == 2:
            mel_spectrogram = mel_spectrogram.unsqueeze(0)
        assert mel_spectrogram.dim() == 3, mel_spectrogram.dim()
        waveform = self.vocoder(mel_spectrogram)  # ts[B,163872]
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        waveform = waveform[:, :self.original_waveform_length]
        waveform = waveform.cpu().float()
        return waveform  # ts[B,163872]

    @torch.no_grad()
    def ddim_noising(  # ts[B, C:8, lT:256, lM:16] -> ts[B, C:8, lT:256, lM:16]
        self,
        latents: torch.Tensor,
        num_inference_steps: int = 50,
        transfer_strength: int = 1,
    ):

        device = latents.device

        # DDIM 전용 Scheduler로 세팅
        old_offset = self.scheduler.config.steps_offset

        self.scheduler.config.steps_offset = 0
        self.scheduler.set_timesteps(num_inference_steps, device=device)  
        all_timesteps = self.scheduler.timesteps  # ts[980, 960, ..., 0] (length: num_inference_steps)
        t_enc = int(transfer_strength * num_inference_steps)
        used_timesteps = all_timesteps[-t_enc:]

        noisy_latents = latents.clone()

        self.scheduler.config.steps_offset = old_offset
        
        noise = torch.randn_like(noisy_latents)
        noisy_latents = self.scheduler.add_noise(noisy_latents, noise, all_timesteps[-t_enc])

        return noisy_latents

    @torch.no_grad()
    def ddim_denoising(  # ts[B, C:8, lT:256, lM:16] -> ts[B, C:8, lT:256, lM:16]
        self,
        latents: torch.Tensor,
        generated_prompt_embeds: torch.Tensor,
        prompt_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        num_inference_steps: int = 50,
        transfer_strength: int = 1,
        guidance_scale: float = 7.5,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: Optional[int] = 1,
    ):
        r"""
        - cross_attention_kwargs (`dict`, optional): cross attention 설정.
        - callback (`Callable`, optional): 특정 step마다 호출할 함수.
        - callback_steps (`int`, default=1): callback 호출 주기.
        Returns:
        - `torch.Tensor`: Denoised latents.
        """

        device = latents.device
        do_cfg = guidance_scale > 1.0
        old_offset = self.scheduler.config.steps_offset

        self.scheduler.config.steps_offset = 0
        self.scheduler.set_timesteps(num_inference_steps, device=device)  
        all_timesteps = self.scheduler.timesteps
        t_enc = int(transfer_strength * num_inference_steps)
        used_timesteps = all_timesteps[-t_enc:]
        
        extra_step_kwargs = self.pipe.prepare_extra_step_kwargs(generator=None, eta=0.0)  # DDIM eta 설정

        num_warmup_steps = len(used_timesteps) - t_enc * self.scheduler.order

        for i, t in enumerate(used_timesteps):
            # expand latents if classifier free guidance
            latent_model_input = (torch.cat([latents] * 2) if do_cfg else latents)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            # predict noise
            noise_pred = self.unet(
                latent_model_input, t,
                encoder_hidden_states=generated_prompt_embeds,
                encoder_hidden_states_1=prompt_embeds,
                encoder_attention_mask_1=attention_mask,
            ).sample

            # guidance
            if do_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # DDIMScheduler의 step
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

            # callback
            if i == len(used_timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(self.scheduler, "order", 1)
                    callback(step_idx, t, latents)

            self.scheduler.config.steps_offset = old_offset

        return latents

    def edit_audio_with_ddim(  # ts[B, 1, T:1024, M:64] -> mel/wav
        self,
        mel: torch.Tensor,
        text: Union[str, List[str]],
        duration: float,
        batch_size: int,
        transfer_strength: float,
        guidance_scale: float,
        ddim_steps: int,
        return_type: str = "ts",  # "ts" or "np" or "mel"
        clipping = False,
    ):
        
        assert self.evalmode, "Let mode be eval"

        # ========== 사전 setting ==========
        # assert get_bit_depth(original_audio_file_path) == 16, \
        #     f"원본 오디오 {original_audio_file_path}의 bit depth는 16이어야 함"

        if duration > self.audio_duration:
            print(f"Warning: 지정한 duration {duration}s가 원본 오디오 길이 {self.audio_duration}s보다 큼")
            # round_up_duration(audio_file_duration)
            # print(f"duration을 {duration}s로 조정")

        # # 재현성을 위한 seed 설정
        # seed_everything(int(seed))

        # ========== mel -> latents ==========
        assert mel.dim() == 4, mel.dim()
        init_latent_x = self.encode_audios(mel)
        
        if torch.max(torch.abs(init_latent_x)) > 1e2:
            init_latent_x = torch.clamp(init_latent_x, min=-10.0, max=10.0)  # clipping

        # ========== DDIM Inversion (noising) ==========
        prompt_embeds, attention_mask, generated_prompt_embeds = self.pipe.encode_prompt(
            prompt=[text]*batch_size, 
            device=self.device, 
            do_classifier_free_guidance=True,
            num_waveforms_per_prompt=1,
            )

        # t_enc step으로 ddim noising
        noisy_latents = self.ddim_noising(
            latents=init_latent_x,
            num_inference_steps=ddim_steps,
            transfer_strength=transfer_strength,
        )
        
        # ========== DDIM Denoising (editing) ==========
        edited_latents = self.ddim_denoising(
            latents=noisy_latents,
            prompt_embeds=prompt_embeds,
            attention_mask=attention_mask,
            generated_prompt_embeds=generated_prompt_embeds,
            num_inference_steps=ddim_steps,
            transfer_strength=transfer_strength,
            guidance_scale=guidance_scale,
        )

        # ========== latent -> waveform ==========
        # mel spectrogram 복원
        mel_spectrogram = self.decode_latents(edited_latents)
        
        # mel clipping은 선택
        if clipping:
            mel_spectrogram = torch.maximum(torch.minimum(mel_spectrogram, mel), mel)

        if return_type == "mel":
            assert mel_spectrogram.shape[-2:] == (1024,64)
            return mel_spectrogram

        # waveform 변환
        edited_waveform = self.mel_to_waveform(mel_spectrogram)

        # duration보다 긴 경우 자르기
        expected_length = int(duration * self.vocoder.config.sampling_rate)  # 원본 samples 수
        assert edited_waveform.ndim == 2, edited_waveform.ndim
        edited_waveform = edited_waveform[:, :expected_length]
        
        # type 결정 ("pt"인 경우에는 torch.Tensor 그대로 반환)
        if return_type == "np":
            edited_waveform = edited_waveform.cpu().numpy()
        else:
            assert return_type == "ts"
        
        return edited_waveform


    def edit(  # ts[B, 1, T:1024, M:64] -> mel/wav
        self,
        mel: torch.Tensor,
        text: Union[str, List[str]],
        inv_text: Union[str, List[str]],
        duration: float,
        batch_size: int,                            #### <----
        timestep_level: float,
        guidance_scale: float,
        ddim_steps: int,
        return_type: str = "ts",  # "ts" or "np" or "mel"
        clipping = False,
    ):
        assert self.evalmode, "Let mode be eval"
        if duration > self.audio_duration:
            print(f"Warning: 지정한 duration {duration}s가 원본 오디오 길이 {self.audio_duration}s보다 큼")
        # ========== mel -> latents ==========
        assert mel.dim() == 4, mel.dim()
        init_latent_x = self.encode_audios(mel)
        if torch.max(torch.abs(init_latent_x)) > 1e2:
            init_latent_x = torch.clamp(init_latent_x, min=-10.0, max=10.0)  # clipping
        # ========== DDIM Inversion (noising) ==========
        ori_prompt_embeds, ori_attention_mask, ori_generated_prompt_embeds = self.pipe.encode_prompt(
            prompt=[inv_text]*batch_size, 
            device=self.device, 
            do_classifier_free_guidance=True,
            num_waveforms_per_prompt=1,
            )
        prompt_embeds, attention_mask, generated_prompt_embeds = self.pipe.encode_prompt(
            prompt=[text]*batch_size, 
            device=self.device, 
            do_classifier_free_guidance=True,
            num_waveforms_per_prompt=1,
            )
        
        # ddim_inversion
        noisy_latents = self.ddim_inversion(
            start_latents=init_latent_x,
            prompt_embeds=ori_prompt_embeds,
            attention_mask=ori_attention_mask,
            generated_prompt_embeds=ori_generated_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=ddim_steps,
            do_cfg=True,
            transfer_strength=timestep_level,
        )
        # ========== DDIM Denoising (editing) ==========
        # ddim_denoising # ddim_sampling
        edited_latents = self.ddim_denoising(
            latents=noisy_latents,
            prompt_embeds=prompt_embeds,
            attention_mask=attention_mask,
            generated_prompt_embeds=generated_prompt_embeds,
            num_inference_steps=ddim_steps,
            transfer_strength=timestep_level,
            guidance_scale=guidance_scale,
        )
        # ========== latent -> waveform ==========
        # mel spectrogram 복원
        mel_spectrogram = self.decode_latents(edited_latents)
        # mel clipping은 선택
        if clipping:
            mel_spectrogram = torch.maximum(torch.minimum(mel_spectrogram, mel), mel)
        if return_type == "mel":
            assert mel_spectrogram.shape[-2:] == (1024,64)
            return mel_spectrogram
        # waveform 변환
        edited_waveform = self.mel_to_waveform(mel_spectrogram)
        # duration보다 긴 경우 자르기
        expected_length = int(duration * self.vocoder.config.sampling_rate)  # 원본 samples 수
        assert edited_waveform.ndim == 2, edited_waveform.ndim
        edited_waveform = edited_waveform[:, :expected_length]
        # type 결정 ("pt"인 경우에는 torch.Tensor 그대로 반환)
        if return_type == "np":
            edited_waveform = edited_waveform.cpu().numpy()
        else:
            assert return_type == "ts"
        return edited_waveform

    @torch.no_grad()
    def ddim_inversion(
        self,
        start_latents,
        prompt_embeds,
        attention_mask,
        generated_prompt_embeds,
        guidance_scale,
        num_inference_steps,
        do_cfg,
        transfer_strength,
    ):  

        start_timestep = int(transfer_strength * num_inference_steps)
        latents = start_latents.clone()
        self.scheduler.set_timesteps(num_inference_steps, device=start_latents.device)
        # Reversed timesteps <<<<<<<<<<<<<<<<<<<<
        timesteps = reversed(self.scheduler.timesteps)
        for i in range(1, num_inference_steps): # range(1, num_inference_steps):
            if i >= start_timestep: continue
            t = timesteps[i]
            # print(t)
            # Expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            noise_pred = self.unet(
                latent_model_input, t,
                encoder_hidden_states=generated_prompt_embeds,
                encoder_hidden_states_1=prompt_embeds,
                encoder_attention_mask_1=attention_mask,
            ).sample
            # Perform guidance
            if do_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            current_t = max(0, t.item() - (1000//num_inference_steps)) # t # max(0, t.item() - (1000//num_inference_steps))
            next_t = t # min(999, t.item() + (1000//num_inference_steps))   # t
            alpha_t = self.scheduler.alphas_cumprod[current_t]
            alpha_t_next = self.scheduler.alphas_cumprod[next_t]
            # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
            latents = (latents - (1-alpha_t).sqrt()*noise_pred)*(alpha_t_next.sqrt()/alpha_t.sqrt()) + (1-alpha_t_next).sqrt()*noise_pred
        return latents

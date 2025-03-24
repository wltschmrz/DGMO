import os
import json
import inspect
from typing import Any, Callable, Dict, List, Optional, Union
import torch
import torch.nn as nn
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDIMScheduler,
)
from diffusers.image_processor import VaeImageProcessor
from transformers import (
    CLIPImageProcessor,
    ClapAudioModelWithProjection,
    ClapProcessor,
    AutoTokenizer,
)
from huggingface_hub import snapshot_download
from .auffusion_utils.auffusion_converter import denormalize_spectrogram
from .auffusion_utils.auffusion_functions import (
    encode_audio_prompt,
    ConditionAdapter,
    import_model_class_from_model_name_or_path,
    retrieve_latents,
)

# Suppress partial model loading warning
os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")

class Auffusion(nn.Module):
    def __init__(self, device='cuda', ckpt="auffusion/auffusion-full", config=None):
        super().__init__()
        self.device = torch.device(device)

        pretrained_model_name_or_path=ckpt
        if not os.path.isdir(pretrained_model_name_or_path):
            pretrained_model_name_or_path = snapshot_download(pretrained_model_name_or_path) 

        self.vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae").to(device=self.device)
        self.unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet").to(device=self.device)
        self.feature_extractor = CLIPImageProcessor.from_pretrained(pretrained_model_name_or_path, subfolder="feature_extractor")
        self.scheduler = DDIMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")

        self.text_encoder_list, self.tokenizer_list, self.adapter_list = [], [], []
        
        condition_json_path = os.path.join(pretrained_model_name_or_path, "condition_config.json")
        self.condition_json_list = json.loads(open(condition_json_path).read())
        
        for i, condition_item in enumerate(self.condition_json_list):
            
            # Load Condition Adapter
            text_encoder_path = os.path.join(pretrained_model_name_or_path, condition_item["text_encoder_name"])
            tokenizer = AutoTokenizer.from_pretrained(text_encoder_path)
            self.tokenizer_list.append(tokenizer)
            text_encoder_cls = import_model_class_from_model_name_or_path(text_encoder_path)
            text_encoder = text_encoder_cls.from_pretrained(text_encoder_path).to(device=self.device)
            self.text_encoder_list.append(text_encoder)
            print(f"LOADING CONDITION ENCODER {i}")

            # Load Condition Adapter
            adapter_path = os.path.join(pretrained_model_name_or_path, condition_item["condition_adapter_name"])
            adapter = ConditionAdapter.from_pretrained(adapter_path).to(device=self.device)
            self.adapter_list.append(adapter)
            print(f"LOADING CONDITION ADAPTER {i}")

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        self.evalmode = True
        self.checkpoint_path = ckpt
        self.audio_duration = 10.24 if not config else config['duration']
        self.original_waveform_length = 10.24 * 16000 #= 163840 # int(self.audio_duration * self.vocoder.config.sampling_rate)  # 10.24 * 16000 = 163840

        print(f'[INFO] audioldm.py: loaded AudioLDM!')

        self.clap_audio_model = ClapAudioModelWithProjection.from_pretrained("laion/clap-htsat-fused")
        self.clap_processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")

    def eval_(self):
        self.evalmode = True

    def train_(self):
        self.evalmode = False

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

    def prepare_extra_step_kwargs(self, generator, eta):
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta
        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

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
        prompt_embeds: torch.Tensor,
        num_inference_steps: int = 50,
        transfer_strength: int = 1,
        guidance_scale: float = 7.5,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: Optional[int] = 1,
    ):
        device = latents.device
        do_cfg = guidance_scale > 1.0
        old_offset = self.scheduler.config.steps_offset

        self.scheduler.config.steps_offset = 0
        self.scheduler.set_timesteps(num_inference_steps, device=device)  
        all_timesteps = self.scheduler.timesteps
        t_enc = int(transfer_strength * num_inference_steps)
        used_timesteps = all_timesteps[-t_enc:]
        
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator=None, eta=0.0)  # DDIM eta 설정

        num_warmup_steps = len(used_timesteps) - t_enc * self.scheduler.order

        for i, t in enumerate(used_timesteps):
            # expand latents if classifier free guidance
            latent_model_input = (torch.cat([latents] * 2) if do_cfg else latents)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict noise
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=None,
                return_dict=False,
            )[0]

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
        prompt_embeds = self.encode_prompt(prompts=text, batch_size=batch_size, do_cfg=True)
        uncond_embeds, cond_embeds = prompt_embeds.chunk(2)

        # t_enc step으로 ddim noising
        noisy_latents = self.ddim_noising(
            latents=init_latent_x,
            num_inference_steps=ddim_steps,
            transfer_strength=transfer_strength,
        )
        
        # ========== DDIM Denoising (editing) ==========
        edited_latents = self.ddim_denoising(
            latents=noisy_latents,
            prompt_embeds=torch.cat([uncond_embeds, cond_embeds]),
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

    def edit_audio_with_ddim_inversion_sampling(  # ts[B, 1, T:1024, M:64] -> mel/wav
        self,
        mel: torch.Tensor,
        original_text: Union[str, List[str]],
        text: Union[str, List[str]],
        duration: float,
        batch_size: int,                            #### <----
        transfer_strength: float,
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
        image = self.image_processor.preprocess(mel)  # 대략 [1, C, H, W] 형태 반환 가정

        # VAE 인코딩 -> latents
        vae_output = self.vae.encode(image)
        audio_latent = retrieve_latents(vae_output)
        init_latent_x = self.vae.config.scaling_factor * audio_latent
        # if torch.max(torch.abs(init_latent_x)) > 1e2:
        #     init_latent_x = torch.clamp(init_latent_x, min=-10.0, max=10.0)  # clipping

        # ========== DDIM Inversion (noising) ==========
        
        cond_audio_text_embed = encode_audio_prompt(
            text_encoder_list=self.text_encoder_list,
            tokenizer_list=self.tokenizer_list,
            adapter_list=self.adapter_list,
            tokenizer_model_max_length=77,
            dtype=image.dtype,
            prompt=[text]*batch_size,
            device=self.device
        )

        uncond_audio_text_embed = torch.zeros_like(cond_audio_text_embed).to(dtype=cond_audio_text_embed.dtype, device=self.device)

        # ddim_inversion
        noisy_latents = self.ddim_inversion(
            start_latents=init_latent_x,
            final_prompt_embeds=uncond_audio_text_embed, #
            guidance_scale=guidance_scale,
            num_inference_steps=ddim_steps,
            do_cfg=False,
            transfer_strength=transfer_strength,
        )

        # ========== DDIM Denoising (editing) ==========
        # ddim_denoising # ddim_sampling
        edited_latents = self.ddim_denoising(
            latents=noisy_latents,
            prompt_embeds=torch.cat([uncond_audio_text_embed, cond_audio_text_embed]),
            num_inference_steps=ddim_steps,
            transfer_strength=transfer_strength,
            guidance_scale=guidance_scale,
        )

        # ========== latent -> waveform ==========
        # mel spectrogram 복원
        # mel_spectrogram = self.decode_latents(edited_latents)

        image = self.vae.decode(edited_latents / self.vae.config.scaling_factor, return_dict=False)[0]
        do_denormalize = [True] * image.shape[0]
        image = self.image_processor.postprocess(image, output_type="pt", do_denormalize=do_denormalize)
        mel_spectrogram_list=[]
        for img in image:
            mel_spec = denormalize_spectrogram(img)
            mel_spec = mel_spec.unsqueeze(0).unsqueeze(0)
            mel_spectrogram_list.append(mel_spec)
        mel_spectrogram = torch.cat(mel_spectrogram_list, dim=0)

        # mel clipping은 선택
        if clipping:
            mel_spectrogram = torch.maximum(torch.minimum(mel_spectrogram, mel), mel)
        if return_type == "mel":
            assert mel_spectrogram.shape[-2:] == (256,1024) # 64
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
        final_prompt_embeds,
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
                latent_model_input,
                t,
                encoder_hidden_states=final_prompt_embeds,
                cross_attention_kwargs=None,
                return_dict=False,
            )[0]
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


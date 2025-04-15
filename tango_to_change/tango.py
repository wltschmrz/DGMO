import json
import torch
from tqdm import tqdm
from huggingface_hub import snapshot_download
from models_ import AudioDiffusion, DDPMScheduler, DDIMScheduler
from audioldm.audio.stft import TacotronSTFT
from audioldm.variational_autoencoder import AutoencoderKL

class Tango:
    def __init__(self, name="declare-lab/tango", device="cuda:1"):
        self.device = device
        path = snapshot_download(repo_id=name)
        
        vae_config = json.load(open("{}/vae_config.json".format(path)))
        stft_config = json.load(open("{}/stft_config.json".format(path)))
        main_config = json.load(open("{}/main_config.json".format(path)))
        
        self.vae = AutoencoderKL(**vae_config).to(device)
        self.stft = TacotronSTFT(**stft_config).to(device)
        self.model = AudioDiffusion(**main_config).to(device)
        
        vae_weights = torch.load("{}/pytorch_model_vae.bin".format(path), map_location=device)
        stft_weights = torch.load("{}/pytorch_model_stft.bin".format(path), map_location=device)
        main_weights = torch.load("{}/pytorch_model_main.bin".format(path), map_location=device)
        
        self.vae.load_state_dict(vae_weights)
        self.stft.load_state_dict(stft_weights)
        self.model.load_state_dict(main_weights)

        print ("Successfully loaded checkpoint from:", name)
        
        self.vae.eval()
        self.stft.eval()
        self.model.eval()
        
        # scheduler = DDPMScheduler.from_pretrained(main_config["scheduler_name"], subfolder="scheduler")
        self.scheduler = DDIMScheduler.from_pretrained(main_config["scheduler_name"], subfolder="scheduler")

        # self.scheduler = DDIMScheduler.from_config(scheduler.config)

    def chunks(self, lst, n):
        """ Yield successive n-sized chunks from a list. """
        for i in range(0, len(lst), n):
            yield lst[i:i + n]
        
    def generate(self, prompt, steps=100, guidance=3, samples=1, disable_progress=True):
        """ Genrate audio for a single prompt string. """
        with torch.no_grad():
            latents = self.model.inference([prompt], self.scheduler, steps, guidance, samples, disable_progress=disable_progress)
            print(latents.shape)
            mel = self.vae.decode_first_stage(latents)
            print(mel.shape)
            wave = self.vae.decode_to_waveform(mel)
        return wave[0], mel
    
    def generate_for_batch(self, prompts, steps=100, guidance=3, samples=1, batch_size=8, disable_progress=True):
        """ Genrate audio for a list of prompt strings. """
        outputs = []
        for k in tqdm(range(0, len(prompts), batch_size)):
            batch = prompts[k: k+batch_size]
            with torch.no_grad():
                latents = self.model.inference(batch, self.scheduler, steps, guidance, samples, disable_progress=disable_progress)
                mel = self.vae.decode_first_stage(latents)
                wave = self.vae.decode_to_waveform(mel)
                outputs += [item for item in wave]
        if samples == 1:
            return outputs
        else:
            return list(self.chunks(outputs, samples))
    
    # @torch.no_grad()
    # def sample(  # ts[B, C:8, lT:256, lM:16] -> ts[B, C:8, lT:256, lM:16]
    #     self,
    #     latents: torch.Tensor,
    #     prompt_embeds: torch.Tensor,
    #     num_inference_steps: int = 50,
    #     noise_level: int = 1,
    #     guidance_scale: float = 7.5,
    # ):
    #     device = latents.device
    #     do_cfg = guidance_scale > 1.0
    #     old_offset = self.scheduler.config.steps_offset
    #     prompt_emb, boolean_prompt_mask = prompt_embeds

    #     self.scheduler.config.steps_offset = 0
    #     self.scheduler.set_timesteps(num_inference_steps, device=device)  
    #     all_timesteps = self.scheduler.timesteps
    #     t_enc = int(noise_level * num_inference_steps)
    #     used_timesteps = all_timesteps[-t_enc:]

    #     for i, t in enumerate(used_timesteps):
    #         # expand latents if classifier free guidance
    #         latent_model_input = (torch.cat([latents] * 2) if do_cfg else latents)
    #         latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

    #         noise_pred = self.model.unet(
    #             latent_model_input, t, encoder_hidden_states=prompt_emb,
    #             encoder_attention_mask=boolean_prompt_mask
    #         ).sample

    #         # guidance
    #         if do_cfg:
    #             noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    #             noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    #         # DDIMScheduler의 step
    #         latents = self.scheduler.step(noise_pred, t, latents).prev_sample

    #         self.scheduler.config.steps_offset = old_offset

    #     return latents

    # @torch.no_grad()
    # def invert(  # ts[B, C:8, lT:256, lM:16] -> ts[B, C:8, lT:256, lM:16]
    #     self,
    #     start_latents,
    #     prompt_embeds,
    #     guidance_scale,
    #     num_inference_steps,
    #     noise_level,
    #     do_cfg=True,
    # ):
    #     prompt_emb, boolean_prompt_mask = prompt_embeds
    #     start_timestep = int(noise_level * num_inference_steps)
    #     latents = start_latents.clone()
    #     self.scheduler.set_timesteps(num_inference_steps, device=start_latents.device)
    #     # Reversed timesteps <<<<<<<<<<<<<<<<<<<<
    #     timesteps = reversed(self.scheduler.timesteps)
    #     for i in range(1, num_inference_steps): # range(1, num_inference_steps):
    #         if i >= start_timestep: continue
    #         t = timesteps[i]
    #         # print(t)
    #         # Expand the latents if we are doing classifier free guidance
    #         latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
    #         latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
    #         noise_pred = self.model.unet(
    #             latent_model_input, t, encoder_hidden_states=prompt_emb,
    #             encoder_attention_mask=boolean_prompt_mask
    #         ).sample
    #         # Perform guidance
    #         if do_cfg:
    #             noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    #             noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    #         current_t = max(0, t.item() - (1000//num_inference_steps)) # t # max(0, t.item() - (1000//num_inference_steps))
    #         next_t = t # min(999, t.item() + (1000//num_inference_steps))   # t
    #         alpha_t = self.scheduler.alphas_cumprod[current_t]
    #         alpha_t_next = self.scheduler.alphas_cumprod[next_t]
    #         # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
    #         latents = (latents - (1-alpha_t).sqrt()*noise_pred)*(alpha_t_next.sqrt()/alpha_t.sqrt()) + (1-alpha_t_next).sqrt()*noise_pred
    #     return latents

    # # edit_audio_with_ddim_inversion_sampling
    # def edit(  # ts[B, 1, T:1024, M:64] -> mel/wav
    #     self,
    #     mel: torch.Tensor,
    #     inv_text,
    #     text,
    #     ddim_steps: int,
    #     timestep_level: float,
    #     guidance_scale: float,
    #     batch_size: int,                            #### <----
    #     duration: float,
    # ):
    #     if duration > 10.24:
    #         print(f"Warning: 지정한 duration {duration}s가 원본 오디오 길이 {self.audio_duration}s보다 큼")
       
    #     # ========== mel -> latents ==========
    #     with torch.no_grad():
    #         init_latent_x = self.vae.get_first_stage_encoding(self.vae.encode_first_stage(mel))
    #         if torch.max(torch.abs(init_latent_x)) > 1e2:
    #             init_latent_x = torch.clamp(init_latent_x, min=-10.0, max=10.0)  # clipping
    #     # ========== Text Embedding (ori&tar) ==========
    #     ori_prompt_embeds, ori_boolean_prompt_mask = self.model.encode_text_classifier_free(inv_text, num_samples_per_prompt=batch_size)
    #     prompt_embeds, boolean_prompt_mask = self.model.encode_text_classifier_free(text, num_samples_per_prompt=batch_size)
    #     # ========== DDIM Inversion (noising) ==========
    #     noisy_latent = self.invert(
    #         start_latents=init_latent_x,
    #         prompt_embeds=(ori_prompt_embeds, ori_boolean_prompt_mask),
    #         num_inference_steps=ddim_steps,
    #         guidance_scale=guidance_scale,
    #         noise_level=timestep_level,
    #     )
    #     # ========== DDIM Denoising (editing) ==========
    #     edited_latent = self.sample(
    #         latents=noisy_latent,
    #         prompt_embeds=(prompt_embeds, boolean_prompt_mask),
    #         num_inference_steps=ddim_steps,
    #         noise_level=timestep_level,
    #         guidance_scale=guidance_scale,
    #     )
    #     # ========== latent -> mel spectrogram ==========
    #     mel_spectrogram = self.vae.decode_first_stage(edited_latent)
    #     assert mel_spectrogram.shape[-2:] == (1024,64), mel_spectrogram.shape
    #     return mel_spectrogram

    @torch.no_grad()  # ts[B, C:8, lT:256, lM:16] -> ts[B, C:8, lT:256, lM:16]
    def sample(self,
        latents: torch.Tensor,
        prompt_embeds: tuple,
        num_inference_steps: int = 50,
        start_step: int = 0,
        guidance_scale: float = 3,
    ):
        device = latents.device
        do_cfg = guidance_scale > 1.0

        prompt_emb, boolean_prompt_mask = prompt_embeds

        self.scheduler.set_timesteps(num_inference_steps, device=device)  
        timesteps = self.scheduler.timesteps
        print('')
        print("len:",len(timesteps))
        print(timesteps.tolist())

        latents = latents.clone()

        for i in range(start_step, num_inference_steps-1):
            print("i:", i, end=' | ')
            t = self.scheduler.timesteps[i]
            print("t:", t.item(), end=' | ')

            latent_model_input = (torch.cat([latents] * 2) if do_cfg else latents)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            noise_pred = self.model.unet(
                latent_model_input, t, encoder_hidden_states=prompt_emb,
                encoder_attention_mask=boolean_prompt_mask
            ).sample

            if do_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            prev_t = max(1, t.item() - (1000//num_inference_steps)) # t-1
            print('curr: ', t.item(), end=' | ')
            print('prev: ', prev_t)
            alpha_t = self.scheduler.alphas_cumprod[t.item()]
            alpha_t_prev = self.scheduler.alphas_cumprod[prev_t]
            predicted_x0 = (latents - (1-alpha_t).sqrt()*noise_pred) / alpha_t.sqrt()
            direction_pointing_to_xt = (1-alpha_t_prev).sqrt()*noise_pred
            latents = alpha_t_prev.sqrt()*predicted_x0 + direction_pointing_to_xt
        # if self.model.set_from == "pre-trained":
        #     print("pre-trained!!!!!!!!!!!!")
        #     latents = self.model.group_out(latents.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
        return latents

    @torch.no_grad()
    def invert(
        self,
        start_latents: torch.Tensor,
        prompt: tuple,
        num_inference_steps: int = 50,
        guidance_scale: float = 3,
        fin_step=1000,
    ):
        prompt_emb, boolean_prompt_mask = prompt
        
        device = self.device
        do_cfg = guidance_scale > 1.0
        latents = start_latents.clone()

        intermediate_latents = []

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        
        timesteps = self.scheduler.timesteps
        timesteps = reversed(timesteps)  # Reversed timesteps
        print('')
        print("len:",len(timesteps))
        print("steps: ", timesteps.tolist())

        for i in range(1, num_inference_steps):
            if i >= num_inference_steps: continue
            if i > fin_step: continue
            print("i:", i, end=' | ')
            t = timesteps[i]
            print("t:", t.item(), end=' | ') ##
            
            latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
            # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            noise_pred = self.model.unet(
                latent_model_input, t, encoder_hidden_states=prompt_emb,
                encoder_attention_mask=boolean_prompt_mask
            ).sample
            
            if do_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
            current_t = max(0, t.item() - (1000//num_inference_steps))
            next_t = min(1000, t.item())
            print('curr:', current_t, end=' | ')
            print('next:', next_t)
            alpha_t = self.scheduler.alphas_cumprod[current_t]
            alpha_t_next = self.scheduler.alphas_cumprod[next_t]
            # # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
            latents = (latents - (1-alpha_t).sqrt()*noise_pred)*(alpha_t_next.sqrt()/alpha_t.sqrt()) + (1-alpha_t_next).sqrt()*noise_pred

            intermediate_latents.append(latents.unsqueeze(0))

        return torch.cat(intermediate_latents)

    def edit(  # ts[B, 1, T:1024, M:64] -> mel/wav
        self,
        mel,
        inv_text,
        text,
        ddim_steps,
        timestep_level,
        guidance_scale,
        batch_size,
        duration,
    ):
        self.audio_duration = 10.24
        if duration > self.audio_duration:
            print(f"Warning: 지정한 duration {duration}s가 원본 오디오 길이 {self.audio_duration}s보다 큼")
        # assert mel.dim() == 4 and mel.shape[-2:] == (1024,512), (mel.dim(), mel.shape)
        start_step = ddim_steps - int(ddim_steps * timestep_level)
        print(f"\n\n>> start_step: {start_step}, ddim_steps: {ddim_steps}, timestep_level: {timestep_level}")
        # ========== mel -> latents ==========
        with torch.no_grad():
            init_latent_x = self.vae.get_first_stage_encoding(self.vae.encode_first_stage(mel))
            if torch.max(torch.abs(init_latent_x)) > 1e2:
                init_latent_x = torch.clamp(init_latent_x, min=-10.0, max=10.0)  # clipping
        # ========== Text Embedding (ori&tar) ==========
        ori_prompt_embeds, ori_boolean_prompt_mask = self.model.encode_text_classifier_free(inv_text, num_samples_per_prompt=batch_size)
        prompt_embeds, boolean_prompt_mask = self.model.encode_text_classifier_free(text, num_samples_per_prompt=batch_size)
        # ========== DDIM Inversion (noising) ==========
        noisy_latents = self.invert(
            start_latents=init_latent_x,
            prompt=(ori_prompt_embeds, ori_boolean_prompt_mask),
            num_inference_steps=ddim_steps,
            guidance_scale=guidance_scale,
            fin_step=ddim_steps-start_step,
        )
        print('')
        front_idx = ddim_steps-start_step-1
        back_idx = -(start_step)
        print("front index:",front_idx)
        print("back index:",back_idx)
        print(list(noisy_latents.shape))
        # ========== DDIM Denoising (editing) ==========
        edited_latent = self.sample(
            latents=noisy_latents[ddim_steps-start_step-1],
            prompt_embeds=(prompt_embeds, boolean_prompt_mask),
            num_inference_steps=ddim_steps,
            start_step=start_step-1,
            guidance_scale=guidance_scale,
        )
        # ========== latent -> mel spectrogram ==========
        mel_spectrogram = self.vae.decode_first_stage(edited_latent)
        assert mel_spectrogram.shape[-2:] == (1024,64), mel_spectrogram.shape
        return mel_spectrogram

    


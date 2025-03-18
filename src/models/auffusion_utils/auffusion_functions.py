# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import List, Optional, Union

import torch
from diffusers.utils import (
    deprecate,
    logging,
#        randn_tensor,
)
import torch.nn as nn
import os, json, PIL
import numpy as np
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm
from transformers import PretrainedConfig


import torch.nn.functional as F


def json_dump(data_json, json_save_path):
    with open(json_save_path, 'w') as f:
        json.dump(data_json, f, indent=4)
        f.close()


def json_load(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        f.close()
    return data      


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


PipelineImageInput = Union[
    PIL.Image.Image,
    np.ndarray,
    torch.Tensor,
    List[PIL.Image.Image],
    List[np.ndarray],
    List[torch.Tensor],
]
def json_dump(data_json, json_save_path):
    with open(json_save_path, 'w') as f:
        json.dump(data_json, f, indent=4)
        f.close()


def json_load(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        f.close()
    return data                


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    if "t5" in model_class.lower():
        from transformers import T5EncoderModel
        return T5EncoderModel
    if "clap" in model_class.lower():
        from transformers import ClapTextModelWithProjection
        return ClapTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


class ConditionAdapter(nn.Module):
    def __init__(self, config):
        super(ConditionAdapter, self).__init__()
        self.config = config
        self.proj = nn.Linear(self.config["condition_dim"], self.config["cross_attention_dim"])
        self.norm = torch.nn.LayerNorm(self.config["cross_attention_dim"])
        print(f"INITIATED: ConditionAdapter: {self.config}")

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path):
        config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        ckpt_path = os.path.join(pretrained_model_name_or_path, "condition_adapter.pt")
        config = json.loads(open(config_path).read())
        instance = cls(config)
        instance.load_state_dict(torch.load(ckpt_path))
        print(f"LOADED: ConditionAdapter from {pretrained_model_name_or_path}")
        return instance

    def save_pretrained(self, pretrained_model_name_or_path):
        os.makedirs(pretrained_model_name_or_path, exist_ok=True)
        config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        ckpt_path = os.path.join(pretrained_model_name_or_path, "condition_adapter.pt")        
        json_dump(self.config, config_path)
        torch.save(self.state_dict(), ckpt_path)
        print(f"SAVED: ConditionAdapter {self.config['model_name']} to {pretrained_model_name_or_path}")




def prepare_extra_step_kwargs(scheduler, generator, eta):
    # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
    # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
    # and should be between [0, 1]

    accepts_eta = "eta" in set(inspect.signature(scheduler.step).parameters.keys())
    extra_step_kwargs = {}
    if accepts_eta:
        extra_step_kwargs["eta"] = eta

    # check if the scheduler accepts generator
    accepts_generator = "generator" in set(inspect.signature(scheduler.step).parameters.keys())
    if accepts_generator:
        extra_step_kwargs["generator"] = generator
    return extra_step_kwargs
    


   


### vocoder model ###
LRELU_SLOPE = 0.1
MAX_WAV_VALUE = 32768.0


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def get_config(config_path):
    config = json.loads(open(config_path).read())
    config = AttrDict(config)
    return config

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


class ResBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.h = h
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)



class Generator(torch.nn.Module):
    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        # self.conv_pre = weight_norm(Conv1d(80, h.upsample_initial_channel, 7, 1, padding=3))
        self.conv_pre = weight_norm(Conv1d(h.num_mels, h.upsample_initial_channel, 7, 1, padding=3)) # change: 80 --> 512
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            if (k-u) % 2 == 0:
                self.ups.append(weight_norm(
                    ConvTranspose1d(h.upsample_initial_channel//(2**i), h.upsample_initial_channel//(2**(i+1)),
                                    k, u, padding=(k-u)//2)))
            else:
                self.ups.append(weight_norm(
                    ConvTranspose1d(h.upsample_initial_channel//(2**i), h.upsample_initial_channel//(2**(i+1)),
                                    k, u, padding=(k-u)//2+1, output_padding=1)))
            
            # self.ups.append(weight_norm(
            #     ConvTranspose1d(h.upsample_initial_channel//(2**i), h.upsample_initial_channel//(2**(i+1)),
            #                     k, u, padding=(k-u)//2)))
            

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, subfolder=None):
        if subfolder is not None:
            pretrained_model_name_or_path = os.path.join(pretrained_model_name_or_path, subfolder)
        config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        ckpt_path = os.path.join(pretrained_model_name_or_path, "vocoder.pt")

        config = get_config(config_path)
        vocoder = cls(config)

        state_dict_g = torch.load(ckpt_path)
        vocoder.load_state_dict(state_dict_g["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
        return vocoder
    
    @torch.no_grad()
    def inference(self, mels, lengths=None):
        self.eval()
        with torch.no_grad():
            wavs = self(mels).squeeze(1)

        wavs = (wavs.cpu().numpy() * MAX_WAV_VALUE).astype("int16")

        if lengths is not None:
            wavs = wavs[:, :lengths]

        return wavs
    

def encode_audio_prompt(
    text_encoder_list,
    tokenizer_list,
    adapter_list,
    tokenizer_model_max_length,
    dtype,
    prompt,
    device,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    do_classifier_free_guidance=False
):
    """
    Encode textual prompts for the audio modality using multiple text encoders and adapters.
    """
    assert len(text_encoder_list) == len(tokenizer_list), "Mismatched text_encoder and tokenizer counts."
    if adapter_list is not None:
        assert len(text_encoder_list) == len(adapter_list), "Mismatched text_encoder and adapter counts."

    def get_prompt_embeds(prompt_list, device):
        if isinstance(prompt_list, str):
            prompt_list = [prompt_list]

        prompt_embeds_list = []
        for p in prompt_list:
            encoder_hidden_states_list = []
            for j in range(len(text_encoder_list)):
                input_ids = tokenizer_list[j](p, return_tensors="pt", padding=True, truncation=True, max_length=77).input_ids
                input_ids=input_ids.to(device)
                cond_embs = text_encoder_list[j](input_ids).last_hidden_state
                # Pad/truncate embeddings
                if cond_embs.shape[1] < tokenizer_model_max_length:
                    pad_len = tokenizer_model_max_length - cond_embs.shape[1]
                    cond_embs = F.pad(cond_embs, (0, 0, 0, pad_len), value=0)
                else:
                    cond_embs = cond_embs[:, :tokenizer_model_max_length, :]

                if adapter_list is not None:
                    cond_embs = adapter_list[j](cond_embs)
                    encoder_hidden_states_list.append(cond_embs)

            prompt_embeds_batch = torch.cat(encoder_hidden_states_list, dim=1)
            prompt_embeds_list.append(prompt_embeds_batch)

        return torch.cat(prompt_embeds_list, dim=0)

    if prompt_embeds is None:           
        prompt_embeds = get_prompt_embeds(prompt, device)

    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    if do_classifier_free_guidance:
        # Create negative prompt embeddings for classifier-free guidance
        negative_prompt_embeds = torch.zeros_like(prompt_embeds, dtype=prompt_embeds.dtype, device=device)
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    return prompt_embeds



def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")
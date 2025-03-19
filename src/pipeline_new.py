# mixed audio 파일 하나 가져와서 경로랑 텍스트 주면 결과값 주는거
# result/ 에 뭐 줄거임?? >> mixed audio랑 sep의 mel spec이랑, ref, sep의 wav

import os
import sys
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(proj_dir, 'src_audioldm')
sys.path.extend([proj_dir, src_dir])
import torch
import torch.nn as nn
from src.models import AudioLDM, AudioLDM2, Auffusion, Mask
from src.data_processing import AudioDataProcessor
from src.utils import load_config

class DGMO(nn.module):
    def __init__(self, config_path="/configs/DGMO.yaml", *, device=None, **kwargs):
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
                self.ldm = AudioLDM(repo_id=self.repo_id, device=self.device)
                self.channel = 1
            case "audioldm2":
                self.ldm = AudioLDM2(repo_id=self.repo_id, device=self.device)
                self.channel = 1
            case "auffusion":
                self.ldm = Auffusion(repo_id=self.repo_id, device=self.device)
                self.channel = 3
            case _:
                raise ValueError(f"Invalid repo_id: {self.repo_id}")
            
        self.processor = AudioDataProcessor(config_path=ldm_config, device=self.device)
        self.mask = self.init_mask(self.processor, channel=self.channel)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.mask.parameters(), lr=learning_rate)
        self.device = self.ldm.device

        self.iter_sisdrs = []
        self.iter_sdris = []

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

    def init_mask(self, processor, channel=1):
        return Mask(
            channel=channel,
            height=processor.n_freq,
            width=self.target_length,
            device=self.device
            )
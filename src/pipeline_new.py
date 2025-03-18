# mixed audio 파일 하나 가져와서 경로랑 텍스트 주면 결과값 주는거
# result/ 에 뭐 줄거임?? >> mixed audio랑 sep의 mel spec이랑, ref, sep의 wav

import os
import sys
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(proj_dir, 'src_audioldm')
sys.path.extend([proj_dir, src_dir])
import torch
import torch.nn as nn

class DGMO(nn.module):
    def __init__(self, config_path=None, *, device=None, **kwargs):
        super(DGMO, self).__init__()
        self.device = torch.device(device)

        config = load_config(config_path) if config_path else {}
        config.update(kwargs)
        self._apply_config(config)
        
        
        
        
        
        self.ldm = AudioLDM()
        self.mask = Mask()
        self.processor = AudioDataProcessor()
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.mask.parameters(), lr=learning_rate)
        self.device = self.ldm.device
        self.learning_rate = config['learning_rate']
        self.num_epochs = config['num_epochs']
        self.batchsize = config['batchsize']
        self.strength = config['strength']
        self.iteration = config['iteration']
        self.text = config['text']
        self.steps = config['steps']
        self.mixed_text = config['mixed_text'] or None
        self.iter_sisdrs = []
        self.iter_sdris = []

    def _apply_config(self, config):
        for key, value in config.items():
            if isinstance(value, dict):
                self._apply_config(value)
            else:
                setattr(self, key, value)
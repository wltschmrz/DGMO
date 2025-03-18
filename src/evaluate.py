import os
from tqdm import tqdm
import numpy as np
from benchmarks.evaluate_audiocaps import AudioCapsEvaluator
from benchmarks.evaluate_vggsound import VGGSoundEvaluator
from benchmarks.evaluate_music import MUSICEvaluator
from benchmarks.evaluate_esc50 import ESC50Evaluator

from utils import (
    calculate_sdr,
    calculate_sisdr,
    parse_yaml,
    get_mean_sdr_from_dict,
)

def eval(checkpoint_path, config_yaml='config/audiosep_base.yaml'):

    log_dir = 'eval_logs'
    os.makedirs(log_dir, exist_ok=True)

    device = "cuda"
    
    configs = parse_yaml(config_yaml)

    vggsound_evaluator = VGGSoundEvaluator()    # VGGSound+ Evaluator
    audiocaps_evaluator = AudioCapsEvaluator()    # AudioCaps Evaluator
    music_evaluator = MUSICEvaluator()    # MUSIC Evaluator
    esc50_evaluator = ESC50Evaluator()    # ESC-50 Evaluator

    # # Load model
    # query_encoder = CLAP_Encoder().eval()

    pl_model = load_ss_model(
        configs=configs,
        checkpoint_path=checkpoint_path,
        query_encoder=query_encoder
    ).to(device)

    print(f'-------  Start Evaluation  -------')

    # evaluation on VGGSound+ (YAN)
    SISDR, SDRi = vggsound_evaluator(pl_model)
    msg_vgg = "VGGSound Avg SDRi: {:.3f}, SISDR: {:.3f}".format(SDRi, SISDR)
    print(msg_vgg)
    
    # evaluation on MUSIC
    SISDR, SDRi = music_evaluator(pl_model)
    msg_music = "MUSIC Avg SDRi: {:.3f}, SISDR: {:.3f}".format(SDRi, SISDR)
    print(msg_music)

    # evaluation on ESC-50
    SISDR, SDRi = esc50_evaluator(pl_model)
    msg_esc50 = "ESC-50 Avg SDRi: {:.3f}, SISDR: {:.3f}".format(SDRi, SISDR)
    print(msg_esc50)

    # evaluation on AudioCaps
    SISDR, SDRi = audiocaps_evaluator(pl_model)
    msg_audiocaps = "AudioCaps Avg SDRi: {:.3f}, SISDR: {:.3f}".format(SDRi, SISDR)
    print(msg_audiocaps)
    
    msgs = [msg_vgg, msg_audiocaps, msg_music, msg_esc50]

    # open file in write mode
    log_path = os.path.join(log_dir, 'eval_results.txt')
    with open(log_path, 'w') as fp:
        for msg in msgs:
            fp.write(msg + '\n')
    print(f'Eval log is written to {log_path} ...')
    print('-------------------------  Done  ---------------------------')

if __name__ == '__main__':
    eval(checkpoint_path='checkpoint/audiosep_base.ckpt')

   






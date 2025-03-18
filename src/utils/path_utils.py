import os
import sys
from pathlib import Path

# 프로젝트 최상위 디렉토리 자동 탐색
root_dir = Path(__file__).resolve().parent.parent  # 프로젝트 최상위 폴더 기준

# 하위 폴더 경로 설정
config_dir = root_dir / "configs"
data_dir = root_dir / "data"
logs_dir = root_dir / "logs"
src_dir = root_dir / "src"
scripts_dir = root_dir / "scripts"
results_dir = root_dir / "results"

eval_dir = src_dir / "benchmarks"
eval_metadata_dir = eval_dir / "benchmarks/metadata"
processing_dir = src_dir / "data_processing"
model_dir = src_dir / "models"
utils_dir = src_dir / "utils"

sys.path.extend(map(str, [
    root_dir,
    scripts_dir,
    eval_dir,
    eval_metadata_dir,
    processing_dir,
    model_dir,
    utils_dir
    ]))

def get_config_fpath(filename):  # .config dir 내 file path 반환
    return config_dir / filename

def get_data_fpath(filename):  # .data dir 내 file path 반환
    return data_dir / filename

def get_evalcsv_fpath(filename):  # .scripts/evaluation/metadata dir 내 file path 반환
    return eval_metadata_dir / filename

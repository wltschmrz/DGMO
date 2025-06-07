#!/bin/bash

# GPU 설정
export CUDA_VISIBLE_DEVICES=2

# 혼합 오디오 파일 경로와 텍스트 쿼리 입력
MIX_PATH="./data/samples/dog_barking_and_cat meowing.wav"
TEXTS=("dog barking" "cat meowing")

# 설정 파일과 저장 위치
CONFIG_PATH="./configs/DGMO.yaml"
BASE_SAVE_DIR="./results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SAVE_DIR="${BASE_SAVE_DIR}/run_${TIMESTAMP}"
mkdir -p "$SAVE_DIR"

echo "🟨 Mixture: $MIX_PATH"
echo "🟨 Config:  $CONFIG_PATH"
echo "🟨 Output:  $SAVE_DIR"
echo "🟨 Queries:"
for t in "${TEXTS[@]}"; do
  echo "   - $t"
done
echo "-----------------------------------"

# 파일명 자동 생성 & 실행
for text in "${TEXTS[@]}"; do
  fname=$(echo "$text" | tr ' ' '_' | xargs).wav  # 공백 제거 및 파일명 생성
  echo "🔄 Running DGMO for: \"$MIX_PATH\" → \"$SAVE_DIR/$fname\""
  python3 src/pipeline.py \
    --config_path "$CONFIG_PATH" \
    --mix_wav_path "$MIX_PATH" \
    --text "$text" \
    --save_dir "$SAVE_DIR" \
    --save_fname "$fname"
done

echo "✅ Finished. Output saved in: $SAVE_DIR"
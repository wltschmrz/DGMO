import torch
import time

# 특정 GPU 선택 (예: 1번 GPU를 점유하고 싶다면 "cuda:1")
device = torch.device("cuda:3")

# 1GB 크기의 dummy tensor 생성
dummy = torch.randn((800, 800, 800), device=device)

# 무한 루프로 GPU 연산 반복 (연산을 가볍게 유지하면서도 점유)
while True:
    dummy = dummy@dummy  # 의미 없는 연산

    torch.cuda.synchronize()   # 실제로 연산을 GPU에서 수행하게 함
    time.sleep(0.001)            # 과도한 전력 소모 방지
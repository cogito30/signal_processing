import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------
# 1. 딥러닝 파트: 파라미터를 예측하는 신경망 (Controller)
# --------------------------------------------------------
class ParameterPredictor(nn.Module):
    def __init__(self, hidden_dim=32):
        super(ParameterPredictor, self).__init__()
        # 멜 스펙트로그램 같은 특징을 입력받아 파라미터를 추론한다고 가정
        self.net = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2) # [주파수 f, 진폭 A] 2개의 값을 출력!
        )
        
    def forward(self, x):
        out = self.net(x)
        # 주파수와 진폭은 음수가 될 수 없으므로 Softplus 활성화 함수를 통과시킵니다.
        # (ReLU와 비슷하지만 미분이 부드럽게 되는 함수)
        f = torch.nn.functional.softplus(out[:, 0]) 
        A = torch.nn.functional.softplus(out[:, 1])
        return f, A

# --------------------------------------------------------
# 2. 전통 DSP 파트: 미분 가능한 사인파 생성기 (Synthesizer)
# --------------------------------------------------------
class DifferentiableOscillator(nn.Module):
    def __init__(self, sample_rate=16000):
        super(DifferentiableOscillator, self).__init__()
        self.sample_rate = sample_rate

    def forward(self, f, A, duration_samples=1000):
        # 1. 시간 축 만들기 (PyTorch Tensor로 생성!)
        t = torch.linspace(0, duration_samples / self.sample_rate, duration_samples)
        
        # 2. 파동 공식 적용: A * sin(2 * pi * f * t)
        # f와 A는 방금 신경망이 예측한 텐서 값이므로, 이 연산 그래프를 통해 미분이 유지됩니다!
        wave = A * torch.sin(2 * torch.pi * f * t)
        
        return wave

# --------------------------------------------------------
# 3. DDSP 결합 및 실행 테스트
# --------------------------------------------------------
# 가상의 특징 데이터 (배치 1, 특징 128차원)
dummy_feature = torch.randn(1, 128)

# 네트워크와 오실레이터 초기화
predictor = ParameterPredictor()
oscillator = DifferentiableOscillator()

# [실행 흐름]
# 특징 데이터 -> (신경망) -> 주파수, 진폭 예측 -> (오실레이터) -> 소리 배열 출력
predicted_f, predicted_A = predictor(dummy_feature)
generated_audio = oscillator(predicted_f, predicted_A)

print(f"신경망이 예측한 주파수: {predicted_f.item():.2f} Hz")
print(f"신경망이 예측한 진폭: {predicted_A.item():.2f}")
print(f"생성된 오디오 배열 크기: {generated_audio.shape}")

# 시각화
audio_numpy = generated_audio.detach().numpy()
plt.figure(figsize=(10, 3))
plt.plot(audio_numpy, color='#2ecc71', linewidth=2)
plt.title(f"DDSP Generated Wave (f={predicted_f.item():.1f}Hz, A={predicted_A.item():.2f})", fontsize=14)
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.show()
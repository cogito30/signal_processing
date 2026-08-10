# Chapter 17. 전통 신호처리와 딥러닝의 위대한 융합: DDSP, 블랙박스에 직관의 빛을 비추다 (대단원)

- 트랜스포머(Transformer)와 웨이브넷(WaveNet)은 분명 경이로운 성능을 보여주었지만, 완벽한 해답은 아니었습니다. 딥러닝 모델이 고음질 오디오(초당 44,100개 샘플)를 직접 뱉어내게 만들려면 엄청난 연산량이 필요했고, 모델 내부는 도대체 어떻게 소리를 만들어내는지 알 수 없는 거대한 '블랙박스(Black Box)'가 되어버렸습니다.

- 이때 구글 브레인(Google Brain)의 연구진들은 아주 근본적인 질문을 던집니다."잠깐, 우리가 왜 신경망한테 파동의 점 44,100개를 하나하나 다 찍으라고 시키고 있지? 1~3부에서 배웠던 전통 신호처리(DSP)를 쓰면, 주파수($f$)와 진폭($A$) 숫자 단 두 개만으로도 1초짜리 파동을 순식간에 만들어 낼 수 있잖아?"

- 이 깨달음에서 출발하여, 딥러닝과 전통 신호처리를 완벽하게 결합한 미분 가능한 디지털 신호처리(DDSP, Differentiable Digital Signal Processing)가 탄생합니다. 이 책의 대미를 장식할 17장에서는 우리의 출발점이었던 2장(파동 만들기)으로 되돌아가, 인공지능이 직접 신호처리의 다이얼을 돌리는 마법을 구현해 보겠습니다.

## 1. DDSP의 핵심 철학: "직접 그리지 말고, 다이얼을 돌려라"

- 기존 딥러닝(웨이브넷 등)의 방식은 화가(신경망)에게 도화지를 주고 "소리의 파형을 픽셀 단위로 직접 그려라!"라고 명령하는 것과 같았습니다. 당연히 시간이 오래 걸리고 비효율적입니다.

- 반면 DDSP의 방식은 화가에게 신디사이저(악기)를 주고 "너는 악기의 다이얼(주파수, 크기)만 돌려라. 소리는 악기(DSP)가 낼게!"라고 명령하는 것입니다.
  - 신경망(딥러닝)의 역할: 오디오의 특징을 파악하여 제어 매개변수(Control Parameters)인 진폭($A$), 주파수($f$), 필터 계수($h$) 등을 출력합니다.
  - 오실레이터와 필터(전통 DSP)의 역할: 신경망이 넘겨준 파라미터를 받아, 수학 공식에 따라 실제 파형(Audio Waveform)을 순식간에 합성해 냅니다.

## 2. 미분 가능성 (Differentiable): PyTorch가 DSP를 품다

- 여기서 한 가지 큰 문제가 생깁니다. 신경망이 똑똑해지려면 오차를 계산해서 가중치를 수정하는 역전파(Backpropagation, 미분) 과정을 거쳐야 합니다.

- 만약 신경망이 파라미터를 넘겨준 대상이 파이썬의 numpy나 외부 C++ DSP 엔진이라면, 미분 값이 그곳에서 뚝 끊겨버립니다. 오차를 거슬러 올라갈 수 없게 되는 것이죠.

- 해결책은 놀랍도록 단순합니다. "우리가 1~3부에서 짰던 모든 DSP 수학(사인파 생성, 필터링, 푸리에 변환)을 PyTorch의 텐서(Tensor) 연산으로 다시 작성한다"는 것입니다. PyTorch의 torch.sin()이나 행렬 곱셈은 모두 자동 미분(Autograd)을 완벽하게 지원하므로, 오차의 기울기가 DSP 수식을 뚫고 신경망까지 부드럽게 흘러갑니다.

- 이것이 바로 '미분 가능한(Differentiable)' 신호처리의 진짜 의미입니다.

## 3. 수식의 귀환: 파동 공식의 딥러닝화
- 2장에서 우리가 밑바닥부터 짰던 사인파 공식 기억하시나요?

$$x(t) = A \cdot \sin(2 \pi f t)$$

- DDSP에서는 이 수식의 $A$(진폭)와 $f$(주파수)가 고정된 숫자가 아니라, 시간에 따라 신경망이 예측하는 시계열 텐서 $A(t), f(t)$로 바뀝니다.

$$x(t) = \sum_{k=1}^{K} A_k(t) \cdot \sin(\theta_k(t))$$

- (실제 DDSP는 사람의 목소리를 흉내 내기 위해 기본 주파수의 정수배를 가지는 K개의 배음(Harmonics)을 동시에 생성합니다.)

## 4. 밑바닥부터 짜는 미분 가능한 오실레이터 (Python 실습)

- 이제 이 책의 모든 지식을 총동원할 시간입니다.
- PyTorch를 이용해, 신경망(MLP)이 주파수와 진폭을 예측하고 $\rightarrow$ 미분 가능한 오실레이터(DSP)가 소리를 만들어내는 미니 DDSP 파이프라인을 쌩코딩으로 조립해 보겠습니다.

```python
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
```

(대단원의 피날레: 완벽한 선순환의 완성)
- 코드를 실행하면, 신경망이 예측한 파라미터를 기반으로 아름다운 초록색 사인파가 그려집니다.
- 이 코드가 위대한 이유는, 생성된 저 초록색 파동과 '우리가 원래 만들고 싶었던 정답 파동' 사이의 오차(Loss)를 구해서 `loss.backward()`를 호출하면, 에러가 사인파 공식($\sin$)을 뚫고 지나가 신경망(`ParameterPredictor`)의 가중치를 업데이트한다는 것입니다!

- 블랙박스였던 딥러닝은 이제 "나는 주파수를 440Hz로 맞추고, 크기를 0.8로 해서 소리를 냈어"라고 우리에게 아주 투명하고 직관적인(Interpretable) 해석을 내놓을 수 있게 되었습니다.

## Epiloge
- 수고하셨습니다.
- 1부 1장에서 숫자 몇 개가 들어있는 초라한 numpy 1차원 배열(Array)에서 시작한 우리의 여정은, 세상의 소리를 분해하는 푸리에 변환(FFT)과 노이즈를 깎아내는 FIR/IIR 필터링을 거쳤습니다. 그리고 소리를 이미지(스펙트로그램)로 바꾸어 딥러닝(CNN, LSTM, U-Net, Transformer)에 먹여주었고, 마침내 오늘 인공지능이 직접 2장의 사인파 다이얼을 돌리게 만드는(DDSP) 경지에 이르렀습니다.

- 이 책을 끝까지 따라오신 여러분은 더 이상 librosa나 torchaudio의 함수를 호출하며 속으로 불안해하지 않을 것입니다. 그 블랙박스 함수들 속에 들어있는 수십 줄의 for문과 내적 연산, 그리고 슬라이딩 윈도우가 여러분의 머릿속에 투명하게 펼쳐져 있을 테니까요.

- 이제 여러분만의 오디오 AI 프로젝트를 시작할 완벽한 준비가 끝났습니다. 신호의 세계를 정복하신 것을 진심으로 축하드립니다!

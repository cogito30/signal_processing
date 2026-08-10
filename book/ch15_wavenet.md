# Chapter 15. 멀리 있는 신호까지 바라보기: 수만 개의 파동을 엮어내는 WaveNet과 팽창 합성곱

- 5부까지 우리는 존재하는 신호의 특징을 뽑아내고(CNN, LSTM) 노이즈를 지우는(U-Net) 방법을 배웠습니다. 이제 딥러닝 신호처리의 가장 경이로운 단계이자 최종 진화형인 '오디오 생성(Audio Generation)'의 세계로 들어갑니다.

- 컴퓨터가 글자를 읽어주는 TTS(Text-to-Speech)나 없는 음악을 만들어내는 AI를 상상해 봅시다. 사람의 목소리를 아주 자연스럽게 만들어내려면, 기계는 앞서 배운 나이퀴스트 정리에 따라 1초당 무려 16,000개에서 44,100개의 점(데이터)을 순차적으로 찍어내야 합니다.

- 이 엄청난 계산량을 기존의 RNN이나 일반 CNN으로 감당하려다 보니 속도는 처참하게 느렸고, 생성된 목소리는 로봇처럼 뚝뚝 끊겼습니다. 2016년, 구글 딥마인드(DeepMind)는 이 한계를 완벽하게 박살 내는 혁명적인 모델 웨이브넷(WaveNet)을 발표하며 오디오 생성의 역사를 새로 씁니다. 이번 장에서는 그 혁명의 심장인 팽창 합성곱(Dilated Convolution)의 비밀을 코드로 파헤쳐 보겠습니다.

## 1. 시야(Receptive Field)의 한계와 1D CNN의 절망

- 음성을 생성할 때 가장 중요한 것은 '맥락(Context)'입니다.
- "안-녕-하-세-[?]" 다음에 "요"가 올지 다른 단어가 올지 맞추려면, 모델은 아주 먼 과거의 단어(안)까지 기억하고 바라볼 수 있어야 합니다. 딥러닝에서 모델이 한 번에 바라볼 수 있는 데이터의 범위를 수용 영역(Receptive Field)이라고 부릅니다.

- 12장에서 배운 기본 1D CNN을 떠올려 봅시다.
- 크기가 3인 필터(`kernel_size=3`)를 사용하면, 한 번에 고작 3개의 샘플(과거 2개 + 현재 1개)만 볼 수 있습니다. 1초(16,000개 샘플) 전의 소리까지 바라보려면, 이 필터를 무려 8,000층(Layers)이나 쌓아야 합니다! 컴퓨터 메모리가 버틸 수 없는 미친 짓이죠.

- 그렇다고 필터 크기를 16,000으로 무식하게 키우면 연산량이 폭발해 버립니다.

## 2. 아이디어 1: 듬성듬성 보며 시야를 넓히다 (Dilated Convolution)

- 이 딜레마를 해결하기 위해 구글 연구진은 천재적인 꼼수를 냅니다.
- "필터 크기를 키우지 말고, 필터의 간격을 벌려서 듬성듬성 보자!"

- 이것이 바로 팽창 합성곱(Dilated Convolution)입니다.
  - Layer 1 (Dilation=1): 바로 옆에 붙어있는 데이터 3개를 봅니다. (간격 1)
  - Layer 2 (Dilation=2): 한 칸씩 건너뛰며 데이터 3개를 봅니다. (간격 2)
  - Layer 3 (Dilation=4): 세 칸씩 건너뛰며 데이터 3개를 봅니다. (간격 4)
  - ...
  - Layer 10 (Dilation=512): 511칸씩 건너뛰며 데이터를 봅니다.

- 간격(Dilation)을 $1, 2, 4, 8, 16 \dots$ 식으로 지수함수적으로(Exponentially) 늘려나가면 어떻게 될까요?
- 층(Layer)은 고작 10층만 쌓았을 뿐인데, 모델의 시야(Receptive Field)는 무려 1,024칸으로 폭발적으로 넓어집니다! 연산량은 크기가 3인 기본 필터와 완벽하게 똑같은데 말이죠.

## 3. 아이디어 2: 미래를 훔쳐보지 마라 (Causal Convolution)

- 오디오를 생성할 때 지켜야 할 철칙이 하나 더 있습니다.
- 우리는 지금 과거의 소리를 바탕으로 '다음 순간'의 파동을 예측하고 있습니다. 그런데 필터가 오른쪽(미래)의 데이터를 참고해버리면, 정답을 미리 훔쳐보는 부정행위(Data Leakage)가 됩니다.

- 따라서 윈도우가 미끄러질 때, 필터는 오직 '현재'와 '과거(왼쪽)'의 데이터만 바라보도록 강제해야 합니다. 이를 인과적 합성곱(Causal Convolution)이라고 부릅니다.

- 코드로 구현하는 방법은 아주 허무할 정도로 간단합니다. 배열의 왼쪽(과거)에만 패딩(0)을 잔뜩 채워 넣고, 합성곱을 수행하면 자연스럽게 미래를 볼 수 없게 됩니다.

## 4. 밑바닥부터 짜는 WaveNet 블록 (Python 실습)

- 이제 파이토치(PyTorch)를 이용해, 미래를 보지 않으면서(Causal) 시야를 기하급수적으로 넓히는(Dilated) 웨이브넷의 핵심 부품인 Causal Dilated Conv1d 모듈을 직접 조립해 보겠습니다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalDilatedConv1d(nn.Module):
    """
    미래의 데이터를 보지 않고(Causal), 
    간격을 벌려 시야를 넓히는(Dilated) WaveNet의 핵심 블록입니다.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(CausalDilatedConv1d, self).__init__()
        
        # 1. 패딩 크기 계산
        # 미래(오른쪽)를 보지 않기 위해 왼쪽(과거)으로만 패딩을 추가합니다.
        # 간격(dilation)이 벌어질수록 덮어야 할 패딩 크기도 커집니다.
        self.left_padding = (kernel_size - 1) * dilation
        
        # 2. 1D CNN 레이어 정의
        # PyTorch의 내장 Conv1d에 dilation 옵션만 켜주면 됩니다!
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, x):
        # x의 형태: [배치 사이즈, 채널, 신호 길이]
        
        # 3. 비대칭 패딩(Asymmetric Padding) 적용
        # F.pad(텐서, (왼쪽 패딩, 오른쪽 패딩))
        # 미래를 보지 못하도록 오른쪽에 추가될 패딩은 0으로 설정합니다.
        x_padded = F.pad(x, (self.left_padding, 0))
        
        # 4. 합성곱 통과
        return self.conv(x_padded)

# --------------------------------------------------------
# 시야(Receptive Field)가 넓어지는 마법 테스트하기
# --------------------------------------------------------
# 테스트 신호: 길이 10짜리 가상의 오디오 텐서
test_signal = torch.arange(1, 11, dtype=torch.float32).view(1, 1, 10)
print(f"원본 신호: {test_signal.squeeze().numpy()}")

# Dilation을 1, 2, 4로 늘려가며 레이어 생성 (필터 크기는 모두 2)
layer_d1 = CausalDilatedConv1d(in_channels=1, out_channels=1, kernel_size=2, dilation=1)
layer_d2 = CausalDilatedConv1d(in_channels=1, out_channels=1, kernel_size=2, dilation=2)
layer_d4 = CausalDilatedConv1d(in_channels=1, out_channels=1, kernel_size=2, dilation=4)

# 가중치를 1로, 편향(bias)을 0으로 고정하여 덧셈기로 만듭니다 (결과 관찰을 위해)
with torch.no_grad():
    layer_d1.conv.weight.fill_(1.0); layer_d1.conv.bias.fill_(0.0)
    layer_d2.conv.weight.fill_(1.0); layer_d2.conv.bias.fill_(0.0)
    layer_d4.conv.weight.fill_(1.0); layer_d4.conv.bias.fill_(0.0)

# 통과시켜 봅니다!
out_1 = layer_d1(test_signal)
out_2 = layer_d2(out_1)
out_4 = layer_d4(out_2)

# 출력 크기를 보면, 왼쪽으로만 패딩을 넣었기 때문에 
# 계속해서 원래 길이(10)가 완벽하게 유지되는 것을 볼 수 있습니다!
print(f"출력 신호 크기: {out_4.shape} (길이 유지 증명!)")
```

(아키텍처)
- 위 코드가 WaveNet의 전부라고 해도 과언이 아닙니다. `dilation` 변수를 `1, 2, 4, 8, 16, 32...`로 지수함수처럼 늘려가며 위에서 만든 `CausalDilatedConv1d` 블록을 차곡차곡 쌓아 올리면, 모델은 단 몇 개의 층만으로도 과거 몇 초 단위의 방대한 신호를 전부 훑어보면서 다음 파동의 값을 기가 막히게 예측(생성)해 냅니다.

- 이 팽창 합성곱 덕분에, 로봇 같았던 AI의 목소리가 비로소 인간처럼 호흡하고 억양을 가지는 기적이 일어났습니다.

## Summary

- 전통적인 1D CNN은 긴 오디오의 맥락을 파악하기엔 시야(Receptive Field)가 너무 좁다.
- 팽창 합성곱(Dilated Convolution)은 필터의 간격을 띄워서 듬성듬성 보는 방식으로, 연산량 증가 없이 모델의 시야를 기하급수적으로 넓힌다.
- 데이터를 생성할 때 미래를 훔쳐보지 않게 만들기 위해 배열의 왼쪽(과거)에만 패딩을 채워 넣는 기법을 인과적 합성곱(Causal Convolution)이라 한다.

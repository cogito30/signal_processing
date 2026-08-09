# Chapter 07. 합성곱(Convolution)의 진정한 의미: 슬라이딩 윈도우와 메아리

- 2부까지 우리는 신호를 분해하여 그 안에 어떤 주파수가 들어있는지 '관찰'하는 방법을 배웠습니다. 이제 3부부터는 한 걸음 더 나아가, 신호를 우리가 원하는 대로 조작하고 다듬는(Filtering) 기술을 배웁니다. 지글거리는 노이즈를 깎아내거나, 건조한 목소리에 촉촉한 메아리(Echo)를 넣는 작업들이 모두 여기에 속합니다.

- 그리고 이 모든 신호 조작의 중심에는 신호처리 수학의 꽃이자 딥러닝(CNN)의 핵심인 합성곱(Convolution, 컨볼루션)이 자리 잡고 있습니다.

- 수학자들은 합성곱을 복잡한 수식으로 표현하지만, 프로그래머인 우리의 눈으로 보면 이것은 그저 '배열 위를 미끄러지며 내적(Dot Product)을 수행하는 슬라이딩 윈도우'일 뿐입니다. 이번 장에서는 합성곱의 진짜 의미를 해부하고 직접 코드로 구현해 보겠습니다.

## 1. 메아리 방(Echo Room)의 비밀

- 합성곱을 가장 직관적으로 이해할 수 있는 비유는 '메아리(Echo)'입니다.
- 당신이 텅 빈 동굴 속에서 "야!" 하고 짧고 굵게 소리를 쳤다고 상상해 봅시다.

- 0초: "야!" (원본 소리, 가장 큼)
- 1초 뒤: "야.." (벽에 한 번 튕겨서 돌아온 소리, 약간 작아짐)
- 2초 뒤: "야...." (두 번 튕겨서 돌아온 소리, 아주 작아짐)

- 이 동굴이라는 공간은 들어온 소리를 특정한 패턴(점점 줄어드는 3번의 메아리)으로 변형시키는 시스템(System)입니다. 신호처리에서는 이런 시스템의 고유한 특성을 '임펄스 응답(Impulse Response)'이라고 부르며, 보통 $h[n]$이라는 짧은 배열로 표현합니다.

- 예를 들어 위 동굴의 $h[n]$은 `[1.0, 0.5, 0.25]`라는 배열로 쓸 수 있습니다.

- 그렇다면, 이 동굴 안에서 "야!"라는 단발성 소리가 아니라 "안-녕-하-세-요"라는 긴 문장을 말하면 어떻게 될까요? "안"의 메아리와 "녕"의 메아리가 서로 겹치고 뒤섞이게 될 것입니다.

- 이처럼 **원본 신호($x[n]$)**가 **시스템의 특성($h[n]$)** 을 통과하면서 서로 뒤섞이고 융합되어 새로운 결과물($y[n]$)을 만들어내는 과정을 수학적으로 계산하는 것, 그것이 바로 **합성곱(Convolution)** 입니다.


## 2. 수식 해부: 뒤집고, 미끄러지고, 곱해서 더해라!

- 전공 책에 나오는 이산 합성곱의 수식은 다음과 같습니다.

$$y[n] = x[n] * h[n] = \sum_{k=-\infty}^{\infty} x[k] \cdot h[n-k]$$


- 여기서 별표($*$)는 단순 곱하기가 아니라 합성곱 기호입니다. 오른쪽의 시그마 수식을 프로그래머의 언어로 차근차근 번역해 봅시다.

1. $h[k]$: 필터(시스템 특성) 배열입니다.
2. $h[-k]$: 필터 배열을 좌우로 홀라당 뒤집습니다 (Flip).
3. $h[n-k]$: 뒤집은 필터를 오른쪽으로 $n$칸만큼 이동시킵니다 (Shift).
4. $\sum (x \cdot h)$: 원본 신호와 이동시킨 필터가 겹치는 부분끼리 곱하고 모두 더합니다 (내적, Dot Product!)

- 즉, 합성곱이란 필터를 뒤집은 다음, 원본 배열 위를 한 칸씩 미끄러지며(Sliding) 계속해서 내적을 구하는 노가다 작업입니다.

## 3. 밑바닥부터 짜는 1D Convolution (Python 실습)
- 원리를 알았으니 `numpy.convolve` 같은 내장 함수는 쳐다보지도 않고, 오직 for문과 배열 슬라이싱만으로 합성곱을 쌩코딩해 보겠습니다.

```python
import numpy as np
import matplotlib.pyplot as plt

def my_convolve1d(signal, filter_kernel):
    """
    1차원 신호(signal)와 필터(filter_kernel)의 합성곱을 계산합니다.
    """
    N = len(signal)
    M = len(filter_kernel)
    
    # 1. 결과 배열의 크기 계산
    # 신호가 필터를 통과하며 꼬리(메아리)가 남기 때문에, 결과는 두 길이의 합 - 1 이 됩니다.
    result_len = N + M - 1
    result = np.zeros(result_len)
    
    # 2. 필터 뒤집기 (Flip)
    # 파이썬 슬라이싱 [::-1]을 사용하면 배열을 아주 쉽게 뒤집을 수 있습니다.
    flipped_filter = filter_kernel[::-1]
    
    # 3. 원본 신호 앞뒤로 패딩(Padding) 붙이기
    # 필터가 배열 바깥으로 삐져나갈 때 에러가 나지 않도록 앞뒤에 0을 채워줍니다.
    padded_signal = np.pad(signal, (M - 1, M - 1), mode='constant')
    
    # 4. 슬라이딩 윈도우 (미끄러지며 내적하기)
    for i in range(result_len):
        # 겹치는 부분 추출 (윈도우 사이즈 M만큼)
        window = padded_signal[i : i + M]
        
        # 겹치는 부분과 뒤집힌 필터의 내적(Dot Product)을 구해 결과 배열에 쏙!
        result[i] = np.sum(window * flipped_filter)
        
    return result
```

- 코드가 아주 명료합니다! 뒤집고(`[::-1]`), 패딩을 대고(np.pad), 미끄러지면서 겹치는 부분끼리 내적(`np.sum`)하는 것이 전부입니다.


## 4. 합성곱으로 노이즈 깎아내기 (이동 평균 필터)

- 우리가 만든 함수가 얼마나 강력한지 눈으로 확인해 볼 차례입니다.
- 센서에서 측정한 데이터에 지글거리는 노이즈가 잔뜩 끼어있다고 가정해 봅시다. 이 노이즈를 부드럽게 깎아내기 위해, 신호처리에서 가장 널리 쓰이는 기초 필터인 '이동 평균 필터 (Moving Average Filter)'를 합성곱으로 적용해 보겠습니다.

- **이동 평균 필터의 아이디어**: "주변 데이터 5개의 평균을 내서 현재 값을 대체하자!"
- **필터 배열($h[n]$)**: [0.2, 0.2, 0.2, 0.2, 0.2] (5개 값의 합이 1이 되도록 맞춤)

```python
# 1. 가상의 노이즈 낀 신호 만들기
np.random.seed(0)
t = np.linspace(0, 10, 100)
clean_signal = np.sin(t)                         # 원본 사인파
noise = np.random.normal(0, 0.3, len(t))         # 지글거리는 가우시안 노이즈
noisy_signal = clean_signal + noise              # 노이즈가 낀 실제 측정 데이터

# 2. 이동 평균 필터 (Moving Average Filter) 설계
window_size = 5
# 크기가 5이고 모든 원소가 1/5 인 배열 생성 -> [0.2, 0.2, 0.2, 0.2, 0.2]
ma_filter = np.ones(window_size) / window_size 

# 3. 합성곱(Convolution) 실행! 우리가 만든 함수를 통과시킵니다.
smoothed_signal = my_convolve1d(noisy_signal, ma_filter)

# 4. 결과 시각화
plt.figure(figsize=(12, 5))

# 원본 노이즈 신호 (회색)
plt.plot(noisy_signal, color='lightgray', label='Noisy Signal', marker='o', markersize=4)

# 필터를 통과한 부드러운 신호 (빨간색 선)
# (참고: 합성곱을 하면 배열 길이가 늘어나므로, 그래프를 맞추기 위해 앞부분만 자릅니다)
valid_length = len(noisy_signal)
plt.plot(smoothed_signal[window_size//2 : valid_length + window_size//2], 
         color='#e74c3c', linewidth=3, label='Smoothed (Convolved) Signal')

plt.title("Smoothing Noise with Convolution (Moving Average)", fontsize=14)
plt.xlabel("Sample Index")
plt.legend()
plt.tight_layout()
plt.show()
```

(그래프 해석)
- 코드를 실행하면 배경에 뾰족뾰족하게 튀어 있던 회색 노이즈 데이터들 위로, 그 노이즈를 아주 부드럽게 깎아내며 원본 사인파의 형태를 예쁘게 복원해 낸 빨간색 선이 나타납니다.

- 단순히 `[0.2, 0.2, 0.2, 0.2, 0.2]`라는 배열 하나를 합성곱으로 미끄러뜨렸을 뿐인데, 데이터의 불필요한 고주파 노이즈가 말끔히 제거된 것입니다!

## Summary
- 합성곱(Convolution)은 원본 신호가 어떤 시스템(필터)을 통과했을 때 어떻게 변하는지를 계산하는 과정이다.
- 수학적으로는 복잡해 보이지만, 코드로 짜면 "필터를 뒤집어서 원본 배열 위를 미끄러지며 내적을 구하는 것"에 불과하다.
- 합성곱 연산 하나만으로 에코(Echo)를 넣거나 노이즈를 깎아내는 등 신호를 자유자재로 다듬을 수 있다.

이제 우리는 슬라이딩 윈도우(합성곱)를 이용해 신호를 다듬는 원리를 깨우쳤습니다. 하지만 이동 평균 필터는 너무 원시적입니다. 주파수를 정밀하게 칼질하려면 더 정교한 필터가 필요합니다.

# Chapter 09. 피드백이 있는 필터: 과거를 기억하며 연산량을 줄이는 마법 (IIR 필터)

- 8장에서 만든 FIR(Finite Impulse Response) 필터는 원하는 주파수만 정밀하게 잘라내는 훌륭한 메스였습니다. 하지만 이 메스에는 치명적인 단점이 하나 있습니다. 필터의 경계를 더 칼같이 깎아내고 싶을수록, 필터 배열의 길이(`num_taps`)를 100개, 1,000개로 무식하게 늘려야 한다는 점입니다. 배열이 길어지면 7장에서 배운 합성곱(Convolution) 연산량이 기하급수적으로 늘어나 스마트폰 같은 소형 기기에서는 배터리가 남아나질 않게 됩니다.

- 이 문제를 해결하기 위해 공학자들은 아주 기발한 아이디어를 떠올렸습니다.
- "매번 들어오는 입력 신호($x$)만 가지고 계산할 게 아니라, 방금 전에 내가 계산했던 출력 결괏값($y$)을 다시 가져와서(재활용해서) 써먹으면 어떨까?"

- 이것이 바로 피드백(Feedback)의 개념이며, 이 피드백을 활용해 단 몇 개의 배열만으로도 무한한 메아리를 만들어내는 효율의 극치, **IIR(Infinite Impulse Response) 필터** 의 핵심입니다.

## 1. 피드백(Feedback)과 무한한 메아리
- 노래방에서 마이크를 스피커 가까이 가져갔을 때 "삐이익-!" 하는 찢어지는 소리(하울링)를 들어보신 적이 있나요?

- 스피커에서 나온 소리(출력)가 다시 마이크(입력)로 들어가고, 그 소리가 또 증폭되어 스피커로 나오는 무한 루프에 빠졌기 때문입니다. 피드백은 이처럼 '출력이 다시 입력으로 돌아가는 구조'를 말합니다.

- FIR 필터는 피드백이 없습니다. 7장에서 배운 동굴의 메아리($h[n]$)가 배열 길이만큼만 튕기고 딱 끝납니다. (그래서 이름이 Finite, 유한합니다.)

- 하지만 IIR 필터는 방금 계산한 결과를 계속 꼬리표처럼 달고 다닙니다. 0.5배씩 작아지는 피드백을 걸어주면, 소리가 $0.5 \rightarrow 0.25 \rightarrow 0.125 \dots$ 식으로 무한히(Infinite) 이어지며 부드러운 꼬리를 남기게 됩니다.


## 2. 수식 해부: 차분 방정식 (Difference Equation)

- 이 피드백 구조를 수학에서는 차분 방정식이라는 이름으로 부릅니다. 전공 책을 펴면 항상 나오는 IIR 필터의 뼈대 수식입니다.

$$y[n] = \frac{1}{a_0} \left( \sum_{i=0}^{M} b_i \cdot x[n-i] - \sum_{j=1}^{N} a_j \cdot y[n-j] \right)$$

- 수식이 길어 보이지만, 쫄 필요 전혀 없습니다! 우리가 이미 아는 두 가지 덩어리가 합쳐졌을 뿐입니다.

1. **앞부분 ($\sum b \cdot x$)**: 현재와 과거의 '입력값($x$)'들을 섞는 부분입니다. 놀랍게도 FIR 필터(합성곱)와 완벽하게 똑같은 작업입니다! 이때 곱해지는 계수를 배열 $b$ (Feedforward)라고 부릅니다.
2. **뒷부분 ($\sum a \cdot y$)**: 과거의 '출력값($y$)'들을 다시 가져와서 빼주는 부분입니다. 이것이 바로 피드백(Feedback)입니다! 이때 곱해지는 계수를 배열 $a$ 라고 부릅니다.

- 즉, IIR 필터는 $b$ 배열(입력 처리용)과 $a$ 배열(피드백 처리용) 두 가지의 설계도로 작동하는 아주 우아한 필터입니다.

## 3. 밑바닥부터 짜는 IIR 필터 엔진 (Python 실습)

```python
import numpy as np
import matplotlib.pyplot as plt

def my_lfilter(b, a, x):
    """
    배열 b(입력 계수)와 배열 a(피드백 계수)를 이용해 
    입력 신호 x를 필터링하는 IIR 필터 엔진입니다.
    """
    # 결과를 담을 0으로 채워진 빈 배열(출력 y) 생성
    y = np.zeros(len(x))
    
    # 안전장치: 수식에 따라 전체를 a[0]로 나누어 정규화합니다. (보통 a[0]는 1.0입니다)
    b = np.array(b) / a[0]
    a = np.array(a) / a[0]
    
    # 배열의 길이만큼 for문을 돕니다 (시간이 흘러갑니다)
    for n in range(len(x)):
        
        # 1. 입력 처리 (Feedforward: b 배열과 x의 내적)
        feedforward = 0
        for i in range(len(b)):
            if n - i >= 0: # 인덱스가 0보다 클 때만 (과거 데이터가 존재할 때만)
                feedforward += b[i] * x[n - i]
                
        # 2. 피드백 처리 (Feedback: a 배열과 과거의 y 내적)
        feedback = 0
        for j in range(1, len(a)): # a[0]는 이미 나누었으니 j=1부터 시작!
            if n - j >= 0:
                feedback += a[j] * y[n - j]
                
        # 3. 최종 결괏값 y[n] 계산 (입력 - 피드백)
        y[n] = feedforward - feedback
        
    return y
```

- 이 코드야말로 수식을 파이썬으로 1:1 번역한 완벽한 튜토리얼입니다. 현재 시간($n$)에서 과거로 거슬러 올라가며($n-i$, $n-j$), 입력값($x$)과 내가 방금 만든 출력값($y$)을 곱하고 더하는 구조가 한눈에 들어옵니다.

## 4. 단 3개의 숫자로 만드는 강력한 로우패스 필터

- 우리가 만든 엔진이 얼마나 효율적인지 테스트해 볼까요?
- 가장 단순하지만 실무에서 가장 많이 쓰이는 IIR 필터인 **지수 이동 평균(Exponential Moving Average) 필터** 를 설계해 보겠습니다.

- 수식은 아주 단순합니다. "새로 들어온 입력($x$)은 10%만 반영하고, 과거의 내 결괏값($y$)을 90% 유지하자!"
- 이를 차분 방정식 형태로 정리하면 다음과 같습니다.
- $y[n] = 0.1 \cdot x[n] + 0.9 \cdot y[n-1]$
- 이항하면 $\rightarrow$ $y[n] - 0.9 \cdot y[n-1] = 0.1 \cdot x[n]$

- 따라서 우리의 설계도 배열은 딱 3개의 숫자로 끝납니다.
- $b$ 배열: [0.1]$
- a$ 배열: [1.0, -0.9]

- 이 단 3개의 숫자만으로 8장에서 만들었던 101개짜리 FIR 필터와 맞먹는 노이즈 제거 효과를 낼 수 있습니다!

```python
# 1. 노이즈가 잔뜩 낀 테스트 신호 생성
np.random.seed(42)
t = np.linspace(0, 1, 500)
clean_signal = np.sin(2 * np.pi * 3 * t) # 3Hz 원본
noisy_signal = clean_signal + np.random.normal(0, 0.5, len(t)) # 노이즈 추가

# 2. IIR 필터 계수 설계 (위에서 계산한 단 3개의 숫자!)
b_coeffs = [0.1]
a_coeffs = [1.0, -0.9]

# 3. 우리가 만든 엔진으로 필터링 실행
filtered_signal_iir = my_lfilter(b_coeffs, a_coeffs, noisy_signal)

# 4. 결과 시각화
plt.figure(figsize=(12, 5))
plt.plot(t, noisy_signal, color='lightgray', label='Noisy Input')
plt.plot(t, filtered_signal_iir, color='#8e44ad', linewidth=3, label='IIR Filtered (b=[0.1], a=[1, -0.9])')

plt.title("Lightweight IIR Filter (Exponential Smoothing)", fontsize=14)
plt.xlabel("Time (seconds)")
plt.legend()
plt.tight_layout()
plt.show()
```

( 놀라운 결과와 Trade-off )
- 코드를 실행해 보면, 101번을 미끄러지며 곱해야 했던 FIR 필터 대신 단 2번의 곱셈 연산만으로 노이즈가 부드럽게 깎여나간 보라색 선을 볼 수 있습니다. 메모리와 배터리 소비를 50배 넘게 줄인 것입니다!

- 하지만 공짜 점심은 없습니다. 그래프를 자세히 보시면 보라색 선이 원본(회색)보다 살짝 오른쪽으로 밀려(Delay) 있는 것을 볼 수 있습니다. 과거의 값을 계속 끌어다 쓰다 보니, 신호의 반응 속도가 한 박자 늦어지는 부작용(위상 지연, Phase Delay)이 발생하는 것이죠.

## Summary
- IIR 필터는 방금 계산한 출력값($y$)을 다시 입력으로 가져오는 피드백(Feedback) 구조를 가진다.
- FIR 필터가 배열 $b$ 하나만 쓴다면, IIR 필터는 피드백을 다루는 배열 $a$를 추가로 사용한다. (차분 방정식)
- 피드백 덕분에 매우 적은 연산량으로도 강력한 필터링이 가능하지만, 신호가 밀리거나 잘못 설계하면 무한 루프(발산)에 빠질 수 있는 위험이 있다.


# Chapter 02. 파동만들기: 세상의 모든 소리를 빚어내는 재료
- 1장에서 우리는 세상의 물리적인 신호를 파이썬의 1차원 배열(Array)로 담아낼 수 있다는 사실을 알게 되었습니다. 이제 이 배열이라는 텅 빈 캔버스에 직접 신호를 그려볼 차례입니다.

- 가장 먼저 그려볼 신호는 바로 **사인파(Sine Wave)**와 **코사인파(Cosine Wave)**입니다.
- "아니, 고등학교 수학 시간에 날 괴롭히던 그 삼각함수?"라며 뒷걸음질 치고 싶으실지도 모릅니다. 하지만 신호처리에서 이 두 파동은 단순한 수학 공식을 넘어, 세상의 모든 복잡한 소리를 만들어낼 수 있는 '가장 순수한 형태의 레고 블록'과 같습니다.

- 이번 장에서는 복잡한 삼각함수 공식을 암기하는 대신, 직관적인 세 가지 변수만으로 자유자재로 파동을 통제하고 코드로 찍어내는 방법을 배웁니다.

## 1. 파동을 통제하느 세 가지 마법의 다이얼
- 음악을 들을 때 앰프나 이퀄라이저의 노브(Knob)를 돌려 소리를 조절해 본 적이 있나요? 파이썬에서 사인파를 만들 때도 정확히 세 개의 다이얼만 조절하면 어떤 파동이든 만들어낼 수 있습니다.

- 수학적으로 사인파 신호 $x(t)$는 다음 수식으로 표현됩니다.

$$x(t) = A \cdot \sin(2 \cdot \pi \cdot f \cdot t + \phi)$$

- 이 수식을 구성하는 핵심 다이얼 세 가지를 살펴봅시다.

- 다이얼 1: 진폭 (Amplitude, $A$)
  - 의미: 파동이 위아래로 얼마나 크게 요동치는지를 결정합니다.
  - 청각적 체감: '소리의 크기(볼륨)'입니다. $A$가 크면 큰 소리가 나고, 작으면 작은 소리가 납니다.

- 다이얼 2: 주파수 (Frequency, $f$)
  - 의미: 1초 동안 파동이 몇 번 진동하는지를 나타냅니다. 단위는 헤르츠(Hz)를 사용합니다.
    - 예: 1Hz는 1초에 한 번 출렁입니다. 100Hz는 1초에 100번 출렁입니다.
  - 청각적 체감: '소리의 높낮이(음정)'입니다. 주파수가 낮으면 묵직한 베이스 소리가 나고, 주파수가 높으면 칠판을 긁는 듯한 날카로운 소리가 납니다.

- 다이얼 3: 위상 (Phase, $\phi$)
  - 의미: 파동이 시작하는 출발점(각도)을 결정합니다. 좌우로 신호를 밀거나 당기는 역할을 합니다.
  - 사인과 코사인의 비밀: 사실 사인파와 코사인파는 똑같이 생긴 쌍둥이입니다. 사인파를 왼쪽으로 90도($\frac{\pi}{2}$)만큼 밀어버리면 그것이 바로 코사인파가 됩니다!


## 2. 밑바닥부터 사인파 생성기 만들기
- 이제 세 가지 다이얼을 코드로 구현해 봅시다. 
- 앞으로 책 전체에서 요긴하게 쓸 **나만의 사인파 생성 함수**를 만들어 보겠습니다.

- 앞으로 모든 수학 기호는 아래와 같이 직관적인 변수명으로 치환될 것입니다.
- $t$ $\rightarrow$ time_array (시간 축 배열)
- $A$ $\rightarrow$ amplitude (진폭)
- $f$ $\rightarrow$ freq (주파수)
- $\phi$ $\rightarrow$ phase (위상)

```python
import numpy as np
import matplotlib.pyplot as plt

def generate_sine_wave(freq, amplitude=1.0, phase=0.0, duration=1.0, sample_rate=1000):
    """
    주어진 파라미터로 이산 사인파(1D 배열)를 생성합니다.
    
    Args:
        freq (float): 주파수 (Hz) - 1초에 진동하는 횟수
        amplitude (float): 진폭 - 소리의 크기
        phase (float): 위상 (라디안) - 시작 지점
        duration (float): 신호의 길이 (초)
        sample_rate (int): 1초당 샘플을 추출하는 횟수 (해상도)
        
    Returns:
        t (ndarray): 시간 축 배열
        signal (ndarray): 생성된 신호 배열
    """
    # 1. 시간 축 만들기
    # 0초부터 duration(초)까지, 초당 sample_rate 개수만큼 점을 찍습니다.
    # 예: duration이 2초, sample_rate가 1000이면 총 2000개의 배열 생성
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # 2. 수식을 그대로 코드로 번역! 
    # x(t) = A * sin(2 * pi * f * t + phi)
    signal = amplitude * np.sin(2 * np.pi * freq * t + phase)
    
    return t, signal
```

- 넘파이 배열 `t`에 스칼라값(`freq`, `pi` 등)을 곱하고 `np.sin()`을 씌우는 이 한 줄의 코드는, 루프(for문) 없이도 배열 내부의 1,000개 데이터 전부에 순식간에 사인 연산을 수행합니다. 이것이 바로 넘파이의 강력한 기능인 **브로드캐스팅(Broadcasting)**입니다.

## 3. 다이얼 돌려보기: 진폭, 주파수, 위상 시각화
- 우리가 만든 함수가 제대로 작동하는지, 다이얼을 하나씩 돌려가며 눈으로 확인해 보겠습니다.

```python
# 3개의 그래프를 나란히 비교하기 위한 세팅
fig, axes = plt.subplots(3, 1, figsize=(10, 8))
plt.tight_layout(pad=4.0)

# --- [비교 1: 진폭 (Amplitude) 다이얼] ---
# 기본 5Hz 파동과 진폭을 2배로 키운 5Hz 파동 비교
t1, sig_base = generate_sine_wave(freq=5, amplitude=1.0)
t1, sig_amp2 = generate_sine_wave(freq=5, amplitude=2.0)

axes[0].plot(t1, sig_base, label="Amplitude = 1.0 (Base)", color="gray", linestyle="--")
axes[0].plot(t1, sig_amp2, label="Amplitude = 2.0 (Loud)", color="#3498db")
axes[0].set_title("1. Changing Amplitude (Volume)")
axes[0].legend()

# --- [비교 2: 주파수 (Frequency) 다이얼] ---
# 5Hz(저음) 파동과 10Hz(고음) 파동 비교
t2, sig_freq5 = generate_sine_wave(freq=5, amplitude=1.0)
t2, sig_freq10 = generate_sine_wave(freq=10, amplitude=1.0)

axes[1].plot(t2, sig_freq5, label="Freq = 5 Hz (Bass)", color="gray", linestyle="--")
axes[1].plot(t2, sig_freq10, label="Freq = 10 Hz (Treble)", color="#e74c3c")
axes[1].set_title("2. Changing Frequency (Pitch)")
axes[1].legend()

# --- [비교 3: 위상 (Phase) 다이얼 - 사인과 코사인] ---
# 일반 사인파(위상 0)와 90도(pi/2) 당겨진 파동(코사인) 비교
t3, sig_sine = generate_sine_wave(freq=5, phase=0)
t3, sig_cosine = generate_sine_wave(freq=5, phase=np.pi/2) # 90도 밀기

axes[2].plot(t3, sig_sine, label="Sine (Phase = 0)", color="gray", linestyle="--")
axes[2].plot(t3, sig_cosine, label="Cosine (Phase = $\pi/2$)", color="#2ecc71")
axes[2].set_title("3. Changing Phase (Shift)")
axes[2].legend()

plt.show()
```

- 그래프를 확인해 보면 우리가 의도한 대로 완벽하게 통제되고 있음을 알 수 있습니다.
1. 진폭 변경: 위아래로 더 높이 치솟습니다 (소리가 커짐).
2. 주파수 변경: 1초 안에 파도치는 횟수가 5번에서 10번으로 더 빽빽해집니다 (음정이 높아짐).
3. 위상 변경: 그래프가 왼쪽으로 스윽 이동합니다. 시작점이 0이 아닌 꼭대기(1.0)에서 시작하는 초록색 선, 바로 코사인파가 탄생했습니다.

## 4. 파동 합치기(Superposition)
- 이 장의 마지막 핵심입니다. 왜 이렇게 단순한 모양의 사인파에 집착할까요?
자연계의 파동은 놀라운 성질을 가지고 있습니다. "아무리 복잡한 파동이라도, 단순히 여러 개의 사인파를 더하는 것만으로 만들어낼 수 있다"는 사실입니다. (이를 **파동의 중첩**이라고 합니다.)

- 파이썬에서는 배열끼리의 덧셈 기호 `+` 하나로 신호 합성을 끝낼 수 있습니다. 묵직한 베이스 5Hz 파동과, 날카로운 50Hz 파동을 섞어보겠습니다.

```python
# 1. 재료 준비: 5Hz 저주파수(큰 진폭)와 50Hz 고주파수(작은 진폭)
t, wave1 = generate_sine_wave(freq=5, amplitude=1.0)
t, wave2 = generate_sine_wave(freq=50, amplitude=0.3)

# 2. 신호 합성 (배열끼리 그냥 더하면 됩니다!)
mixed_wave = wave1 + wave2

# 3. 결과 확인
plt.figure(figsize=(10, 4))
plt.plot(t, mixed_wave, color="#9b59b6")
plt.title("Mixed Wave (5Hz + 50Hz)")
plt.xlabel("Time (seconds)")
plt.show()
```

- 결과 그래프를 보면, 큰 물결(5Hz)의 뼈대 위에 자잘한 물결(50Hz)이 타고 흐르는 아주 복잡한 형태의 파동이 만들어졌음을 볼 수 있습니다.

- 우리는 방금 이 세상의 모든 복잡한 소리(내 목소리, 피아노 소리)가 결국 '진폭과 주파수가 다른 수많은 사인파들의 덧셈'에 불과하다는 푸리에(Fourier) 이론의 아주 중요한 힌트를 직접 코드로 증명해 냈습니다.

## Summary
- 사인파는 진폭(크기), 주파수(높낮이), 위상(시작점) 세 가지 변수로 통제할 수 있다.
- 수학 공식 $A\sin(2\pi f t + \phi)$는 넘파이 배열 연산 한 줄로 완벽히 번역된다.
- 배열과 배열을 더하는 것만으로 복잡한 신호를 합성할 수 있다.

- 재료 준비는 끝났습니다. 다음 3장에서는 우리의 목소리나 진짜 음악 파일(wav)처럼 눈에 보이지 않는 데이터를 어떻게 파이썬 배열로 가져올 수 있는지(샘플링), 그리고 그 과정에서 반드시 알아야 할 나이퀴스트(Nyquist) 정리에 대해 알아보겠습니다.

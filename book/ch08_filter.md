# Chapter 08. 노이즈 제거하기: 원하는 주파수만 골라내는 정밀한 메스 (FIR 필터)

- 7장에서 우리는 이동 평균(Moving Average) 필터를 합성곱(Convolution)으로 통과시키며 지글거리는 노이즈를 깎아내는 마법을 경험했습니다.

- 사실 여러분은 이미 로우패스 필터(Low-pass Filter, 저역통과 필터)를 하나 만드신 겁니다! 이동 평균은 데이터의 급격한 변화(고주파 노이즈)를 뭉개고, 완만한 변화(저주파 원본 신호)만 남기기 때문이죠.

- 하지만 이동 평균 필터는 '망치'와 같습니다. 신호를 뭉툭하게 깎아내다 보니, 원본 신호의 중요한 디테일까지 함께 뭉개져 버리는 단점이 있습니다. 이번 장에서는 우리가 2부에서 배운 **주파수 도메인(Frequency Domain)** 지식을 활용하여, 외과의사의 '정밀한 메스'처럼 내가 원하는 주파수 대역만 칼같이 잘라내는 **본격적인 FIR(Finite Impulse Response) 디지털 필터** 를 설계해 보겠습니다.

## 1. 주파수 세상에서 필터 그리기 (이상과 현실)
- 우리가 진짜로 원하는 완벽한 로우패스 필터(Low-pass Filter)는 어떤 모습일까요?
- 주파수 세상(스펙트럼)에서 상상해 봅시다.

- "10Hz 이하의 주파수는 그대로 살리고(1.0 곱하기), 10Hz가 넘는 고주파 노이즈는 완벽하게 죽여라(0 곱하기)."

- 이걸 주파수 그래프로 그리면 완벽한 직사각형(네모) 모양이 됩니다.
- 문제는 이 '주파수 세상의 네모 모양'을 다시 '시간 세상의 필터 배열($h[n]$)'로 가져와야 합성곱 연산을 할 수 있다는 점입니다.

- 수학자들은 이미 그 답을 찾아놓았습니다. 주파수 도메인의 직사각형을 역 푸리에 변환(IFFT)하면, 시간 도메인에서는 그 유명한 싱크 함수(Sinc Function)가 튀어나옵니다!

## 2. 물방울의 파문, 싱크(Sinc) 함수

- 싱크 함수는 수식으로 아주 간단합니다.

$$\text{sinc}(x) = \frac{\sin(\pi x)}{\pi x}$$

- 가운데가 가장 높이 솟아 있고, 양옆으로 갈수록 잔물결처럼 출렁이며 줄어드는 모양입니다. 마치 고요한 호수에 물방울을 똑 떨어뜨렸을 때 퍼져나가는 파문(Ripple)과 완벽하게 똑같이 생겼습니다.

- 놀랍게도, 이 싱크 함수 모양으로 필터 배열($h[n]$)을 만들어서 신호와 합성곱(Convolution)을 하면, 완벽한 로우패스 필터가 작동합니다.

## 3. 밑바닥부터 짜는 FIR 로우패스 필터 (Python 실습)
- 복잡한 내장 함수 대신, `numpy`의 사인 함수(`np.sin`)를 이용해 싱크(Sinc) 필터 배열을 직접 찍어내는 함수를 만들어 봅시다.

```python
import numpy as np
import matplotlib.pyplot as plt

def design_lowpass_fir(cutoff_freq, sample_rate, num_taps):
    """
    Sinc 함수를 이용해 FIR 로우패스 필터 배열을 생성합니다.
    
    Args:
        cutoff_freq (float): 잘라낼 기준 주파수 (Hz)
        sample_rate (int): 샘플링 레이트
        num_taps (int): 필터 배열의 길이 (반드시 홀수여야 가운데 중심이 생깁니다)
    """
    # 1. 중심이 0인 인덱스 배열 만들기 (예: num_taps가 51이면 -25 부터 +25 까지)
    n = np.arange(num_taps) - (num_taps - 1) / 2
    
    # 2. 정규화된 컷오프 주파수 (0.0 ~ 0.5 사이의 비율로 변환)
    fc = cutoff_freq / sample_rate
    
    # 3. 싱크(Sinc) 함수 적용: sin(2 * pi * fc * n) / (pi * n)
    h = np.zeros(num_taps)
    for i, idx in enumerate(n):
        if idx == 0:
            # 분모가 0이 되는 중심(idx=0)은 로피탈의 정리로 2*fc가 됩니다.
            h[i] = 2 * fc 
        else:
            h[i] = np.sin(2 * np.pi * fc * idx) / (np.pi * idx)
            
    # 4. 블랙만(Blackman) 윈도우 적용 (현실적인 타협)
    # 무한히 뻗어가는 싱크 함수를 중간에 싹둑 자르면 파도(Ripple)가 생깁니다.
    # 이를 부드럽게 덮어주기 위해 양끝을 둥글게 눌러주는 윈도우 배열을 곱합니다.
    window = np.blackman(num_taps)
    h_windowed = h * window
    
    return h_windowed
```

( Window란? )
- 무한한 수학의 세계와 달리, 우리의 필터 배열(num_taps) 길이는 유한합니다(예: 51개, 101개). 싱크 함수를 강제로 싹둑 자르면 가장자리에 부작용(에일리어싱)이 생기므로, 양끝을 스르륵 0으로 만들어주는 '블랙만(Blackman)' 같은 덮개(Window)를 곱해주는 것이 디지털 필터 설계의 핵심 테크닉입니다.


## 4. 정밀한 메스로 노이즈 잘라내기 (적용)
- 이제 우리가 직접 만든 `design_lowpass_fir` 필터와 7장에서 만든 `my_convolve1d` 합성곱 함수를 결합해, 고주파 노이즈를 아주 칼같이 도려내 보겠습니다.

- 원본 신호: 2Hz의 예쁜 베이스(저음) 파동
- 노이즈: 50Hz의 날카롭고 지글거리는 고음 파동
- 작전: 컷오프 주파수(Cut-off)를 10Hz로 설정한 로우패스 필터로 50Hz 노이즈만 완벽히 죽여버린다!

```python
# 7장에서 만든 합성곱 함수 (소환)
def my_convolve1d(signal, filter_kernel):
    # (내용은 7장과 동일하므로 생략 - np.convolve로 대체하여 시뮬레이션 가능)
    return np.convolve(signal, filter_kernel, mode='same')

# 1. 2Hz 원본 신호 + 50Hz 노이즈 생성
fs = 1000 # 초당 1000번 샘플링
t = np.linspace(0, 1, fs, endpoint=False)
clean_2hz = np.sin(2 * np.pi * 2 * t)
noise_50hz = 0.5 * np.sin(2 * np.pi * 50 * t)
noisy_signal = clean_2hz + noise_50hz

# 2. FIR 필터 설계 (10Hz 기준으로 자르는 길이 101짜리 정밀 필터)
fir_filter = design_lowpass_fir(cutoff_freq=10, sample_rate=fs, num_taps=101)

# 3. 합성곱(Convolution)으로 필터링 실행!
filtered_signal = my_convolve1d(noisy_signal, fir_filter)

# 4. 결과 시각화
plt.figure(figsize=(12, 6))

# 노이즈가 낀 원본 (회색)
plt.plot(t, noisy_signal, color='lightgray', label='Noisy Signal (2Hz + 50Hz)')

# 우리가 만든 FIR 필터로 정제한 신호 (파란색)
plt.plot(t, filtered_signal, color='#2980b9', linewidth=3, label='Filtered Signal (Only 2Hz)')

plt.title("FIR Low-pass Filter in Action (Cut-off: 10Hz)", fontsize=14)
plt.xlabel("Time (seconds)")
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
```

- 그래프를 확인해 보면 감탄이 나옵니다. 이동 평균 필터(망치)로 깎았을 때는 원본 신호의 진폭도 줄어들고 파형도 찌그러졌지만, 싱크 함수 기반의 FIR 필터(정밀 메스)를 사용하니 지글거리는 50Hz 톱니바퀴 노이즈만 흔적도 없이 사라지고 완벽하게 깨끗한 2Hz 원본 사인파만 살아남았습니다!

## 5. 응용: 하이패스(High-pass) 필터로 뒤집기

- 저음(Low)을 살리는 필터를 만들었으니, 반대로 고주파(High)만 살리는 하이패스 필터는 어떻게 만들까요? 수식을 또 밑바닥부터 짜야 할까요?

- 아닙니다. '모든 소리(전체 주파수)'에서 '저음'을 빼면 당연히 '고음'만 남습니다!
- 이를 **스펙트럼 반전(Spectral Inversion)** 기법이라고 부릅니다.

- 코드로 구현하면 그저 배열의 뺄셈 한 줄이면 끝납니다.

```python
def design_highpass_fir(cutoff_freq, sample_rate, num_taps):
    # 1. 똑같이 로우패스 필터를 만듭니다.
    lowpass = design_lowpass_fir(cutoff_freq, sample_rate, num_taps)
    
    # 2. 모든 소리를 통과시키는 '임펄스(Impulse)' 배열 생성 (가운데만 1, 나머지 0)
    all_pass = np.zeros(num_taps)
    all_pass[num_taps // 2] = 1.0
    
    # 3. (전체 소리) - (저음) = (고음)
    highpass = all_pass - lowpass
    return highpass
```

- 이처럼 배열의 덧셈과 뺄셈만으로 필터의 특성을 자유자재로 뒤집을 수 있는 것이 바로 디지털 신호처리의 매력입니다.


## Summary
- 이상적인 필터를 주파수 도메인에 그리고 시간 도메인으로 가져오면 **싱크(Sinc) 함수** 형태의 배열이 된다.

- **FIR 필터**는 이 싱크 함수 배열을 원본 신호와 합성곱(Convolution)하여 특정 주파수만 정밀하게 잘라내는 기술이다.

- (전체)에서 (로우패스 필터) 배열을 빼는 단순한 연산만으로 **하이패스 필터**를 쉽게 뚝딱 만들 수 있다.
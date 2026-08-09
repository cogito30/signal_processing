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
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
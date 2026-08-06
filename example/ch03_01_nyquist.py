import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------
# 1. 아날로그 원본 신호 (10Hz)
# --------------------------------------------------------
# 완벽한 아날로그 신호를 가정하기 위해 1초에 1000번(1000Hz) 촘촘하게 샘플링합니다.
t_analog = np.linspace(0, 1, 1000, endpoint=False)
signal_analog = np.sin(2 * np.pi * 10 * t_analog) # 10번 출렁임

# --------------------------------------------------------
# 2. 성능이 안 좋은 카메라로 샘플링 (12Hz)
# --------------------------------------------------------
# 1초에 12번만 점을 찍습니다. (fs = 12)
t_discrete = np.linspace(0, 1, 12, endpoint=False)
signal_discrete = np.sin(2 * np.pi * 10 * t_discrete) 

# --------------------------------------------------------
# 3. 앨리어싱 시각화
# --------------------------------------------------------
plt.figure(figsize=(10, 5))

# 원본 10Hz 신호를 흐리게 그립니다.
plt.plot(t_analog, signal_analog, color='lightgray', label='Original Analog (10Hz)')

# 12Hz로 캡처한 디지털 데이터를 빨간 점으로 찍습니다.
plt.plot(t_discrete, signal_discrete, 'ro', markersize=8, label='Sampled Points (fs=12Hz)')

# 캡처된 빨간 점들만 이어붙여 봅니다 (컴퓨터가 인식하는 최종 신호)
plt.plot(t_discrete, signal_discrete, 'r--', label='What Computer Sees (Fake 2Hz)')

plt.title("Aliasing Effect: Fast Wave looks like a Slow Wave!")
plt.xlabel("Time (seconds)")
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
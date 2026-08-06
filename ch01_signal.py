import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------
# 1. 시간 축(Time Axis) 만들기
# --------------------------------------------------------
# 연속 신호를 흉내 내기 위해 0초부터 1초까지 1000개의 점으로 아주 촘촘하게 쪼갭니다.
# 컴퓨터는 완벽한 연속을 표현할 수 없으므로, 점이 아주 많으면 연속된 선처럼 보입니다.
t_continuous = np.linspace(0, 1, 1000)

# 이산 신호는 0초부터 1초까지 단 15번만 채집(Sampling)합니다.
t_discrete = np.linspace(0, 1, 15)

# --------------------------------------------------------
# 2. 가상의 신호(주파수가 2Hz인 사인파) 생성
# --------------------------------------------------------
# np.sin()을 사용하여 시간에 따른 진폭(Amplitude)을 계산합니다.
# 2 * np.pi * 2 * t : 1초에 2번 진동(2Hz)하는 파동을 의미합니다. (자세한 건 2장에서 다룹니다!)
signal_continuous = np.sin(2 * np.pi * 2 * t_continuous)
signal_discrete = np.sin(2 * np.pi * 2 * t_discrete)

# --------------------------------------------------------
# 3. 그래프로 신호 시각화하기
# --------------------------------------------------------
plt.figure(figsize=(12, 5))
plt.style.use('seaborn-v0_8-whitegrid') # 깔끔한 그래프 스타일 적용

# [왼쪽 그래프] 아날로그 신호 (Continuous)
plt.subplot(1, 2, 1)
# 연속 신호는 plot()을 사용하여 점들을 선으로 매끄럽게 잇습니다.
plt.plot(t_continuous, signal_continuous, color='#2c3e50', linewidth=2)
plt.title("Analog Signal (Continuous $f(t)$)", fontsize=14)
plt.xlabel("Time (seconds)", fontsize=12)
plt.ylabel("Amplitude", fontsize=12)
plt.axhline(0, color='black', linewidth=0.5) # 중심선(0) 긋기

# [오른쪽 그래프] 디지털 신호 (Discrete)
plt.subplot(1, 2, 2)
# 이산 신호는 점과 점 사이의 데이터가 없음을 명확히 하기 위해 
# 선으로 잇지 않고 막대(stem) 형태로 그립니다.
plt.stem(t_discrete, signal_discrete, basefmt="k-", linefmt="#e74c3c", markerfmt="ro")
plt.title("Digital Signal (Discrete $x[n]$)", fontsize=14)
plt.xlabel("Sample Index ($n$)", fontsize=12)
plt.axhline(0, color='black', linewidth=0.5)

# 간격 조절 후 출력
plt.tight_layout()
plt.show()
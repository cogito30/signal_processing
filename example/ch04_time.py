import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------
# 1. 미스터리 신호 만들기 (스무디)
# --------------------------------------------------------
# 1초 동안 1000번 샘플링
t = np.linspace(0, 1, 1000, endpoint=False)

# 3Hz 신호(진폭 1.0)와 7Hz 신호(진폭 0.5)를 섞습니다.
mystery_signal = 1.0 * np.sin(2 * np.pi * 3 * t) + 0.5 * np.sin(2 * np.pi * 7 * t)

# --------------------------------------------------------
# 2. 주파수 탐지기 (내적 계산 함수)
# --------------------------------------------------------
def probe_frequency(target_signal, test_freq, time_array):
    """미스터리 신호 안에 특정 주파수(test_freq)가 얼마나 들어있는지 내적으로 검사합니다."""
    # 테스트할 주파수를 가진 순수 사인파를 만듭니다.
    test_wave = np.sin(2 * np.pi * test_freq * time_array)
    
    # 두 배열의 내적(Dot Product)을 구합니다. (element-wise 곱셈 후 총합)
    dot_product = np.sum(target_signal * test_wave)
    return dot_product

# --------------------------------------------------------
# 3. 탐지기로 미스터리 신호 찔러보기
# --------------------------------------------------------
print("=== 주파수 성분 검사 (내적 값) ===")
print(f"1Hz 검사 결과: {probe_frequency(mystery_signal, 1, t):.2f}")
print(f"3Hz 검사 결과: {probe_frequency(mystery_signal, 3, t):.2f} (Bingo!)")
print(f"4Hz 검사 결과: {probe_frequency(mystery_signal, 4, t):.2f}")
print(f"7Hz 검사 결과: {probe_frequency(mystery_signal, 7, t):.2f} (Bingo!)")


# 검사할 주파수 목록 (1Hz부터 10Hz까지)
test_frequencies = np.arange(1, 11)
similarity_scores = []

# for문을 돌며 모든 주파수에 대해 내적(유사도)을 계산합니다.
for freq in test_frequencies:
    score = probe_frequency(mystery_signal, freq, t)
    similarity_scores.append(score)

# 결과를 그래프(주파수 스펙트럼)로 그립니다.
plt.figure(figsize=(10, 4))
plt.stem(test_frequencies, similarity_scores, basefmt="k-", linefmt="#3498db", markerfmt="bo")
plt.title("My First Fourier Transform (Frequency Spectrum)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Similarity (Dot Product)")
plt.xticks(test_frequencies)
plt.show()
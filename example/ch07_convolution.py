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


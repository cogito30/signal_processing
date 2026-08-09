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
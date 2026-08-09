import numpy as np
import time

def my_fft(x):
    """
    1차원 배열 x를 입력받아 고속 푸리에 변환(FFT)을 수행하는 재귀 함수입니다.
    (입력 배열의 길이는 반드시 2의 거듭제곱이어야 합니다)
    """
    N = len(x)
    
    # [1] Base Case: 배열의 길이가 1 이하가 되면 쪼개기를 멈추고 자기 자신을 반환합니다.
    if N <= 1:
        return x
    
    # [2] Divide: 배열을 짝수 인덱스와 홀수 인덱스로 분할합니다.
    # x[0::2] -> 0, 2, 4, 6... 번째 원소
    # x[1::2] -> 1, 3, 5, 7... 번째 원소
    even_part = my_fft(x[0::2]) # 짝수 배열에 대해 다시 FFT (재귀)
    odd_part = my_fft(x[1::2])  # 홀수 배열에 대해 다시 FFT (재귀)
    
    # [3] Conquer & Combine: 쪼개서 계산된 결과를 하나로 합칩니다.
    # W는 합칠 때 필요한 복소수 회전 계수(Twiddle Factor)입니다.
    # k는 0부터 N/2 - 1까지의 배열입니다.
    k = np.arange(N // 2)
    W = np.exp(-1j * 2 * np.pi * k / N)
    
    # FFT의 대칭성을 이용해 앞절반(N/2)과 뒤절반(N/2)을 동시에 구합니다.
    # 앞절반: 짝수 + W * 홀수
    # 뒤절반: 짝수 - W * 홀수
    first_half = even_part + W * odd_part
    second_half = even_part - W * odd_part
    
    # 두 배열을 이어 붙여서(concatenate) 최종 N 길이의 배열을 반환합니다.
    return np.concatenate([first_half, second_half])

# 5장에서 만든 my_dft 함수 (비교를 위해 잠시 소환)
def my_dft(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-1j * 2 * np.pi * k * n / N)
    return X

# 4096개의 랜덤 신호 생성
N = 4096 
np.random.seed(42)
test_signal = np.random.random(N)

print(f"데이터 크기: {N}개")
print("-" * 30)

# 1. 기존 DFT 속도 측정
start_time = time.time()
dft_result = my_dft(test_signal)
dft_time = time.time() - start_time
print(f"my_dft 실행 시간: {dft_time:.4f} 초")

# 2. 새로운 FFT 속도 측정
start_time = time.time()
fft_result = my_fft(test_signal)
fft_time = time.time() - start_time
print(f"my_fft 실행 시간: {fft_time:.4f} 초")

# 3. 속도 차이 계산
speedup = dft_time / fft_time
print("-" * 30)
print(f"결론: FFT가 DFT보다 약 {speedup:.1f}배 빠릅니다!")
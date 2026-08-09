import numpy as np
import matplotlib.pyplot as plt

def my_dft(x):
    """
    1차원 배열 x를 입력받아 이산 푸리에 변환(DFT)을 수행합니다.
    (O(N^2)의 순수 이중 for문 구현)
    """
    N = len(x) # 배열의 총 길이 (데이터 개수)
    
    # 결과를 담을 빈 주머니 (쌍끌이 결과를 담아야 하므로 dtype=complex 로 설정)
    X = np.zeros(N, dtype=complex)
    
    # 바깥쪽 루프 (k): 어떤 주파수를 탐지할 것인가?
    for k in range(N):
        
        # 안쪽 루프 (n): 시간에 따른 배열의 원소들을 하나씩 꺼내어 내적!
        for n in range(N):
            # 오일러의 마법: 탐지기의 각도 계산
            # 수식: -j * 2 * pi * k * n / N
            angle = -1j * 2 * np.pi * k * n / N
            
            # 신호(x[n])에 탐지기(exp(angle))를 곱해서 차곡차곡 더한다 (내적)
            X[k] += x[n] * np.exp(angle)
            
    return X

# --------------------------------------------------------
# 테스트: 위상의 함정에 빠졌던 코사인파를 포획해보자!
# --------------------------------------------------------
# 1초 동안 100번 샘플링
t = np.linspace(0, 1, 100, endpoint=False)

# 4장에서는 실패했을, 90도 밀려난 5Hz 코사인파!
mystery_signal = np.cos(2 * np.pi * 5 * t) 

# 우리가 만든 DFT 함수 실행
spectrum_complex = my_dft(mystery_signal)

# 복소수 주머니에서 피타고라스 정리로 진짜 크기(Magnitude)만 꺼내기
# np.abs()가 실수부^2 + 허수부^2에 루트를 씌워줍니다.
spectrum_magnitude = np.abs(spectrum_complex)

# 그래프 시각화 (앞의 절반만 그립니다. 나이퀴스트 정리 때문에 뒤는 대칭입니다)
plt.figure(figsize=(10, 4))
plt.stem(spectrum_magnitude[:50], basefmt="k-", linefmt="#2ecc71", markerfmt="go")
plt.title("My Custom DFT Result (Phase Trap Bypassed!)")
plt.xlabel("Frequency Index (k)")
plt.ylabel("Magnitude")
plt.show()
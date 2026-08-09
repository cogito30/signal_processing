import numpy as np
import matplotlib.pyplot as plt

# 1. Hz <-> Mel 변환 함수 정의
def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700.0)

def mel_to_hz(mel):
    return 700 * (10**(mel / 2595.0) - 1)

def get_filterbanks(num_filters, n_fft, sample_rate):
    """
    인간의 귀를 모방한 멜 필터뱅크 행렬(Matrix)을 생성합니다.
    """
    # 1단계: 최소/최대 주파수를 Mel 스케일로 변환
    low_mel = hz_to_mel(0)
    high_mel = hz_to_mel(sample_rate / 2)
    
    # 2단계: Mel 스케일 위에서 일정한 간격으로 점을 찍음 (num_filters + 2 개)
    mel_points = np.linspace(low_mel, high_mel, num_filters + 2)
    
    # 3단계: 찍은 점들을 다시 실제 주파수(Hz)로 되돌림! -> 고주파로 갈수록 간격이 넓어짐
    hz_points = mel_to_hz(mel_points)
    
    # 4단계: Hz를 FFT 결과의 인덱스(Bin)로 변환
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    
    # 5단계: 텅 빈 필터뱅크 행렬 만들기 (삼각형 개수 x FFT 절반 크기)
    fbank = np.zeros((num_filters, int(np.floor(n_fft / 2 + 1))))
    
    # 6단계: 겹치는 삼각형(Triangle) 만들기
    for m in range(1, num_filters + 1):
        left = bin_points[m - 1]
        center = bin_points[m]
        right = bin_points[m + 1]
        
        # 삼각형 왼쪽 빗면 (올라가는 선)
        for k in range(left, center):
            fbank[m - 1, k] = (k - left) / (center - left)
            
        # 삼각형 오른쪽 빗면 (내려가는 선)
        for k in range(center, right):
            fbank[m - 1, k] = (right - k) / (right - center)
            
    return fbank

# --------------------------------------------------------
# 시각화: 우리가 만든 삼각형 필터들은 어떻게 생겼을까?
# --------------------------------------------------------
n_fft = 2048
sample_rate = 16000
num_filters = 40 # 40개의 삼각형 사용

filterbank_matrix = get_filterbanks(num_filters, n_fft, sample_rate)

plt.figure(figsize=(10, 4))
# 모든 삼각형을 한 캔버스에 그립니다.
for i in range(num_filters):
    plt.plot(filterbank_matrix[i, :])

plt.title("Mel-Filterbank: 40 Overlapping Triangular Filters", fontsize=14)
plt.xlabel("FFT Frequency Bin (Linear)")
plt.ylabel("Filter Amplitude")
plt.xlim(0, int(n_fft/2))
plt.tight_layout()
plt.show()
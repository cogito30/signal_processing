import numpy as np
import matplotlib.pyplot as plt

def my_stft(signal, n_fft=2048, hop_length=512):
    """
    1차원 신호를 입력받아 2차원 스펙트로그램 행렬을 반환하는 함수입니다.
    
    Args:
        signal: 1차원 오디오 배열
        n_fft: 깍둑썰기 할 조각의 크기 (Window Size)
        hop_length: 윈도우가 한 번에 미끄러질 이동 거리 (Overlap을 결정)
    """
    # 1. 조각의 양끝을 부드럽게 깎아줄 둥근 덮개(Hanning Window) 생성
    window = np.hanning(n_fft)
    
    # 2. 2차원 결과를 차곡차곡 쌓을 빈 리스트
    spectrogram = []
    
    # 3. 배열 위를 미끄러지는 슬라이딩 윈도우 (for문)
    # 0부터 신호의 끝까지, hop_length(512) 보폭으로 이동합니다.
    for i in range(0, len(signal) - n_fft, hop_length):
        
        # 1차원 조각(Chunk) 잘라내기
        chunk = signal[i : i + n_fft]
        
        # 둥근 덮개(Window) 씌우기 -> 양끝 데이터가 스르륵 0으로 떨어짐
        windowed_chunk = chunk * window
        
        # 조각에 대해 FFT 수행 (np.fft.rfft는 실수 신호의 중복된 절반을 버려줍니다)
        fft_result = np.fft.rfft(windowed_chunk)
        
        # 복소수에서 크기(Magnitude)만 추출
        magnitude = np.abs(fft_result)
        
        # 리스트에 추가 (한 줄짜리 주파수 성분표가 차곡차곡 쌓임)
        spectrogram.append(magnitude)
        
    # 4. 쌓인 리스트를 2차원 넘파이 행렬로 변환 후 전치(Transpose: T)
    # (시간을 X축, 주파수를 Y축으로 만들기 위해 행과 열을 바꿈)
    spectrogram_matrix = np.array(spectrogram).T
    
    return spectrogram_matrix


# --------------------------------------------------------
# 테스트: 주파수가 점점 올라가는 소리(Chirp 신호) 만들어보기
# --------------------------------------------------------
fs = 16000 # 16kHz 샘플링
t = np.linspace(0, 2, fs * 2) # 2초 길이

# 처음엔 100Hz였다가 끝날 때 4000Hz로 점점 올라가는 새소리 같은 파동
signal_chirp = np.sin(2 * np.pi * (100 + 1950 * t) * t)

# --------------------------------------------------------
# 우리가 만든 STFT 실행 및 데시벨(dB) 변환
# --------------------------------------------------------
# STFT 행렬 획득! (2차원 Array)
spec_matrix = my_stft(signal_chirp, n_fft=1024, hop_length=256)

# 로그를 씌워 데시벨(dB)로 변환 (log10 안이 0이 되면 에러가 나므로 아주 작은 값 1e-10을 더해줌)
spec_db = 20 * np.log10(spec_matrix + 1e-10)

# --------------------------------------------------------
# 2차원 이미지(Spectrogram)로 시각화
# --------------------------------------------------------
plt.figure(figsize=(12, 6))

# pcolormesh나 imshow를 이용해 2차원 행렬을 열화상 카메라 사진처럼 그립니다.
plt.imshow(spec_db, aspect='auto', origin='lower', cmap='magma', 
           extent=[0, 2, 0, fs/2])

plt.colorbar(format='%+2.0f dB', label='Magnitude (dB)')
plt.title("My First Spectrogram (From Scratch!)", fontsize=16)
plt.xlabel("Time (seconds)", fontsize=12)
plt.ylabel("Frequency (Hz)", fontsize=12)
plt.tight_layout()
plt.show()
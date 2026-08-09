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
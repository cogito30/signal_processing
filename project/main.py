import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

# --------------------------------------------------------
# 1. 가상의 ECG 데이터 준비 (정상 박동 + 기저선 출렁임 노이즈)
# --------------------------------------------------------
np.random.seed(42)
t = np.linspace(0, 5, 500) # 5초 동안 측정

# 가상의 심장 박동 (뾰족한 QRS 파형을 사인파의 조합으로 단순 묘사)
normal_ecg = np.sin(2 * np.pi * 1.2 * t) ** 5 
# 환자의 호흡으로 인한 거대한 저주파 출렁임 (0.2Hz)
baseline_wander = 1.5 * np.sin(2 * np.pi * 0.2 * t) 

# 우리가 실제로 센서에서 얻게 될 지저분한 원본 신호
raw_ecg = normal_ecg + baseline_wander

# --------------------------------------------------------
# 2. 전통 DSP: IIR 버터워스 하이패스 필터 적용 (9장 복습)
# --------------------------------------------------------
def apply_highpass_filter(data, cutoff=0.5, fs=100):
    """0.5Hz 이하의 저주파 출렁임을 깎아내는 IIR 필터"""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    # 버터워스 필터의 설계도(b, a 계수) 획득
    b, a = butter(2, normal_cutoff, btype='high', analog=False)
    # 9장에서 직접 만들었던 my_lfilter 와 동일한 엔진 통과!
    clean_data = lfilter(b, a, data)
    return clean_data

clean_ecg = apply_highpass_filter(raw_ecg)

# --------------------------------------------------------
# 시각화: 필터링 효과 확인
# --------------------------------------------------------
plt.figure(figsize=(12, 4))
plt.plot(t, raw_ecg, color='lightgray', label='Raw ECG (with Baseline Wander)')
plt.plot(t, clean_ecg, color='#e74c3c', label='Cleaned ECG (High-pass Filtered)')
plt.title("Step 1: Removing Baseline Wander with IIR Filter")
plt.legend()
plt.tight_layout()
plt.show()

import torch
import torch.nn as nn

class LSTM_Autoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=16):
        super(LSTM_Autoencoder, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        
        # 1. 인코더 (압축)
        # 긴 심박수 배열이 하나의 작은 은닉 상태(Hidden State)로 압축됩니다.
        self.encoder_lstm = nn.LSTM(input_size=n_features, 
                                    hidden_size=embedding_dim, 
                                    batch_first=True)
        
        # 2. 디코더 (복원)
        # 압축된 특징을 다시 원래의 길이(seq_len)만큼 펼쳐서 복원합니다.
        self.decoder_lstm = nn.LSTM(input_size=embedding_dim, 
                                    hidden_size=n_features, 
                                    batch_first=True)
        
    def forward(self, x):
        # [Step 1] 인코더 통과
        # x 형태: [Batch, Sequence_length, Features]
        _, (hidden, _) = self.encoder_lstm(x)
        
        # 좁은 병목(Bottleneck): hidden 상태가 바로 압축된 심장 박동의 본질입니다.
        # 이를 시퀀스 길이만큼 복제하여 디코더의 입력으로 준비합니다.
        # 형태 변환: [1, Batch, Embed] -> [Batch, Seq_Len, Embed]
        hidden_repeated = hidden[-1].unsqueeze(1).repeat(1, self.seq_len, 1)
        
        # [Step 2] 디코더 통과 (원래 신호로 복원)
        reconstructed_x, _ = self.decoder_lstm(hidden_repeated)
        return reconstructed_x

# --------------------------------------------------------
# 1. 테스트 상황 가정
# --------------------------------------------------------
# 정상 데이터는 복원을 아주 잘했다고 가정 (오차가 거의 없음)
reconstructed_normal = clean_ecg + np.random.normal(0, 0.05, len(clean_ecg))

# 비정상 데이터 생성 (3초 부근에서 갑자기 비정상적으로 요동치는 부정맥 발생)
abnormal_ecg = clean_ecg.copy()
abnormal_ecg[300:350] += 2.0 * np.sin(2 * np.pi * 5 * t[300:350]) 

# 모델이 비정상 데이터를 복원하려 시도하지만, '정상 패턴'만 배운 모델은 
# 요동치는 구간을 복원하지 못하고 그냥 밋밋한 정상 파형을 뱉어버림
reconstructed_abnormal = clean_ecg + np.random.normal(0, 0.05, len(clean_ecg))

# --------------------------------------------------------
# 2. 재구성 오차(Reconstruction Error) 계산 및 알람
# --------------------------------------------------------
# 에러(원본 - 복원)의 제곱을 구합니다.
error_normal = (clean_ecg - reconstructed_normal) ** 2
error_abnormal = (abnormal_ecg - reconstructed_abnormal) ** 2

# 이상 탐지 임계값(Threshold) 설정
THRESHOLD = 0.5 

plt.figure(figsize=(12, 5))

# 오차 그래프 그리기
plt.plot(t, error_normal, color='gray', label='Error (Normal Patient)', alpha=0.5)
plt.plot(t, error_abnormal, color='red', label='Error (Abnormal Patient)', linewidth=2)

# 임계값 선 긋기
plt.axhline(THRESHOLD, color='blue', linestyle='--', label='Anomaly Threshold')

plt.title("Step 3: Anomaly Detection via Reconstruction Error")
plt.xlabel("Time (seconds)")
plt.ylabel("MSE (Error)")
plt.legend()
plt.tight_layout()
plt.show()
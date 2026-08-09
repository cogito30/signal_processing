import torch
import torch.nn as nn
import numpy as np

# --------------------------------------------------------
# 1. 시계열(신호) 데이터 준비
# --------------------------------------------------------
# PyTorch RNN/LSTM의 기본 입력 형태는 3차원 텐서입니다: 
# [배치 크기(Batch), 시퀀스 길이(Time steps), 입력 특징 수(Features)]
# 예: 1개의 오디오 샘플, 100번의 시간 흐름, 매 시간마다 1개의 진폭 값
seq_length = 100
t = np.linspace(0, 2*np.pi, seq_length)
signal = np.sin(t)

# (1, 100, 1) 형태의 FloatTensor로 변환
x_tensor = torch.tensor(signal, dtype=torch.float32).view(1, seq_length, 1)

# --------------------------------------------------------
# 2. LSTM 모델 뼈대 세우기
# --------------------------------------------------------
class SignalLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=16):
        super(SignalLSTM, self).__init__()
        
        # LSTM 층: 1개의 입력(진폭)을 받아 16개의 숨겨진 상태(메모리)로 확장합니다.
        # batch_first=True 로 설정하면 입력 텐서의 첫 번째 차원이 배치(Batch)가 됩니다.
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            batch_first=True)
        
        # 16개의 메모리 정보를 다시 1개의 예측된 진폭 값으로 모아주는 선형(Linear) 층
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # 1. LSTM 통과
        # out: 모든 시간(Time steps)에 대한 결과값
        # (h_n, c_n): 가장 마지막 시간의 은닉 상태(Hidden)와 셀 상태(Cell)
        out, (h_n, c_n) = self.lstm(x)
        
        # 2. 최종 결과 예측 (모든 시간에 대해 선형 층 통과)
        predictions = self.fc(out)
        return predictions

# --------------------------------------------------------
# 3. 모델에 신호 통과시키기
# --------------------------------------------------------
model = SignalLSTM()
print("입력 텐서의 형태:", x_tensor.shape) # [1, 100, 1]

# 미분(학습) 없이 순전파(Forward)만 테스트
with torch.no_grad():
    output_tensor = model(x_tensor)
    
print("출력 텐서의 형태:", output_tensor.shape) # [1, 100, 1]
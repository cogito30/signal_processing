import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------
# 1. 테스트용 신호 만들기 (2Hz 원본 + 50Hz 노이즈)
# --------------------------------------------------------
fs = 1000
t = np.linspace(0, 1, fs, endpoint=False)
clean_signal = np.sin(2 * np.pi * 2 * t)
noisy_signal = clean_signal + 0.5 * np.sin(2 * np.pi * 50 * t)

# PyTorch는 numpy 배열 대신 'Tensor(텐서)'라는 자기만의 배열을 씁니다.
# CNN에 넣기 위해 형태를 [배치 크기, 채널 수, 신호 길이] = [1, 1, 1000] 으로 맞춥니다.
tensor_signal = torch.tensor(noisy_signal, dtype=torch.float32).view(1, 1, -1)

# --------------------------------------------------------
# 2. PyTorch 1D CNN 레이어 소환!
# --------------------------------------------------------
# in_channels=1 (오디오 모노), out_channels=1 (출력도 1개)
# kernel_size=101 (우리가 8장에서 만들었던 101개짜리 FIR 필터 길이와 동일)
cnn_layer = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=101, padding='same', bias=False)

# --------------------------------------------------------
# 3. 8장의 FIR 로우패스 필터를 CNN 가중치(Weight)에 강제 주입!
# --------------------------------------------------------
# 8장에서 짰던 Sinc 함수 기반 로우패스 필터 로직 (간략화된 버전)
fc = 10 / fs
n = np.arange(101) - 50
fir_filter_array = np.sinc(2 * fc * n) * np.blackman(101) 
fir_filter_array = fir_filter_array / np.sum(fir_filter_array) # 정규화

# 우리의 수제 필터 배열을 파이토치 텐서로 변환하여 CNN 레이어의 가중치로 덮어씌웁니다.
with torch.no_grad():
    cnn_layer.weight.data = torch.tensor(fir_filter_array, dtype=torch.float32).view(1, 1, 101)

# --------------------------------------------------------
# 4. PyTorch 텐서로 합성곱 필터링 실행!
# --------------------------------------------------------
# cnn_layer()를 호출하는 순간, 내부적으로 우리가 7장에서 짠 슬라이딩 윈도우 연산이 일어납니다.
filtered_tensor = cnn_layer(tensor_signal)

# 결과를 다시 numpy 배열로 꺼내기
filtered_signal_pytorch = filtered_tensor.squeeze().detach().numpy()

# --------------------------------------------------------
# 5. 결과 확인
# --------------------------------------------------------
plt.figure(figsize=(12, 5))
plt.plot(t, noisy_signal, color='lightgray', label='Noisy Signal (Input to PyTorch)')
plt.plot(t, filtered_signal_pytorch, color='#d35400', linewidth=3, label='PyTorch 1D CNN Output')
plt.title("PyTorch Conv1d as a Traditional FIR Filter", fontsize=14)
plt.xlabel("Time (seconds)")
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
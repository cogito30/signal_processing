import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalDilatedConv1d(nn.Module):
    """
    미래의 데이터를 보지 않고(Causal), 
    간격을 벌려 시야를 넓히는(Dilated) WaveNet의 핵심 블록입니다.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(CausalDilatedConv1d, self).__init__()
        
        # 1. 패딩 크기 계산
        # 미래(오른쪽)를 보지 않기 위해 왼쪽(과거)으로만 패딩을 추가합니다.
        # 간격(dilation)이 벌어질수록 덮어야 할 패딩 크기도 커집니다.
        self.left_padding = (kernel_size - 1) * dilation
        
        # 2. 1D CNN 레이어 정의
        # PyTorch의 내장 Conv1d에 dilation 옵션만 켜주면 됩니다!
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, x):
        # x의 형태: [배치 사이즈, 채널, 신호 길이]
        
        # 3. 비대칭 패딩(Asymmetric Padding) 적용
        # F.pad(텐서, (왼쪽 패딩, 오른쪽 패딩))
        # 미래를 보지 못하도록 오른쪽에 추가될 패딩은 0으로 설정합니다.
        x_padded = F.pad(x, (self.left_padding, 0))
        
        # 4. 합성곱 통과
        return self.conv(x_padded)

# --------------------------------------------------------
# 시야(Receptive Field)가 넓어지는 마법 테스트하기
# --------------------------------------------------------
# 테스트 신호: 길이 10짜리 가상의 오디오 텐서
test_signal = torch.arange(1, 11, dtype=torch.float32).view(1, 1, 10)
print(f"원본 신호: {test_signal.squeeze().numpy()}")

# Dilation을 1, 2, 4로 늘려가며 레이어 생성 (필터 크기는 모두 2)
layer_d1 = CausalDilatedConv1d(in_channels=1, out_channels=1, kernel_size=2, dilation=1)
layer_d2 = CausalDilatedConv1d(in_channels=1, out_channels=1, kernel_size=2, dilation=2)
layer_d4 = CausalDilatedConv1d(in_channels=1, out_channels=1, kernel_size=2, dilation=4)

# 가중치를 1로, 편향(bias)을 0으로 고정하여 덧셈기로 만듭니다 (결과 관찰을 위해)
with torch.no_grad():
    layer_d1.conv.weight.fill_(1.0); layer_d1.conv.bias.fill_(0.0)
    layer_d2.conv.weight.fill_(1.0); layer_d2.conv.bias.fill_(0.0)
    layer_d4.conv.weight.fill_(1.0); layer_d4.conv.bias.fill_(0.0)

# 통과시켜 봅니다!
out_1 = layer_d1(test_signal)
out_2 = layer_d2(out_1)
out_4 = layer_d4(out_2)

# 출력 크기를 보면, 왼쪽으로만 패딩을 넣었기 때문에 
# 계속해서 원래 길이(10)가 완벽하게 유지되는 것을 볼 수 있습니다!
print(f"출력 신호 크기: {out_4.shape} (길이 유지 증명!)")
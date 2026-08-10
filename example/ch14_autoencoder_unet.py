import torch
import torch.nn as nn

class MiniUNet1D(nn.Module):
    def __init__(self):
        super(MiniUNet1D, self).__init__()
        
        # ----------------------------------------------------
        # 1. 인코더 (Encoder) : 배열 크기를 반으로 줄임
        # stride=2 옵션이 윈도우를 두 칸씩 뜀뛰기하게 만들어 크기를 절반으로 압축합니다.
        # ----------------------------------------------------
        self.encoder = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=4, stride=2, padding=1)
        
        # ----------------------------------------------------
        # 2. 디코더 (Decoder) : 배열 크기를 다시 두 배로 늘림
        # ConvTranspose1d는 압축된 텐서를 뻥튀기(Up-sampling)하는 특수 레이어입니다.
        # ----------------------------------------------------
        self.decoder = nn.ConvTranspose1d(in_channels=16, out_channels=16, kernel_size=4, stride=2, padding=1)
        
        # ----------------------------------------------------
        # 3. 최종 출력층 (Final Layer)
        # 스킵 커넥션으로 원본(1채널)과 디코더(16채널)가 합쳐져 총 17채널이 들어옵니다.
        # 이를 다시 1채널짜리 깔끔한 소리 배열로 합칩니다.
        # ----------------------------------------------------
        self.final_conv = nn.Conv1d(in_channels=17, out_channels=1, kernel_size=3, padding=1)
        
    def forward(self, x):
        # x의 형태: [배치, 채널(1), 신호 길이(예: 1000)]
        
        # [Step 1] 인코딩: 좁은 병목으로 밀어 넣기
        enc = self.encoder(x)
        enc = torch.relu(enc) # 크기가 500으로 줄어들고 특징은 16개로 늘어남
        
        # [Step 2] 디코딩: 다시 원래 길이로 복원하기
        dec = self.decoder(enc)
        dec = torch.relu(dec) # 크기가 다시 1000으로 복원됨
        
        # [Step 3] 마법의 스킵 커넥션 (Skip Connection)
        # 원본 신호 'x'와 복원된 신호 'dec'를 채널(dim=1) 방향으로 이어 붙입니다.
        # "노이즈가 지워진 뼈대(dec)" + "선명한 원본 디테일(x)" 의 만남!
        merged = torch.cat([x, dec], dim=1) 
        
        # [Step 4] 최종 융합 및 출력
        out = self.final_conv(merged)
        return out

# --------------------------------------------------------
# 텐서 흐름(Shape) 테스트
# --------------------------------------------------------
# 가상의 노이즈 낀 오디오 신호 (배치 1, 모노 채널 1, 샘플 1024개)
noisy_audio = torch.randn(1, 1, 1024)

# 우리가 만든 U-Net에 통과시키기
model = MiniUNet1D()
clean_audio = model(noisy_audio)

print(f"입력 노이즈 신호 크기: {noisy_audio.shape}")
print(f"출력 복원 신호 크기: {clean_audio.shape} (크기 완벽 유지!)")
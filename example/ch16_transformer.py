import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        
        # 원본 신호(특징)를 Q, K, V로 변환해 줄 선형 레이어 (가중치 행렬)
        self.W_q = nn.Linear(embed_size, embed_size, bias=False)
        self.W_k = nn.Linear(embed_size, embed_size, bias=False)
        self.W_v = nn.Linear(embed_size, embed_size, bias=False)
        
    def forward(self, x):
        # x의 형태: [배치, 시간(Time Steps), 특징 개수(Embed Size)]
        
        # 1. 원본 데이터 x를 Q, K, V로 각각 변환
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # 2. Q와 K의 전치행렬(K^T)을 내적(행렬곱)하여 유사도 점수 계산
        # K.transpose(1, 2)는 시간 축과 특징 축을 맞바꿉니다.
        # dot_product 형태: [배치, 시간, 시간] -> 모든 시간대끼리의 관계망(N x N)
        dot_product = Q @ K.transpose(1, 2)
        
        # 3. 스케일링 (수식의 루트 d_k로 나누기 - 값이 너무 커지는 것 방지)
        scaled_dot_product = dot_product / (self.embed_size ** 0.5)
        
        # 4. Softmax를 씌워 0~1 사이의 어텐션 가중치(비율)로 변환
        attention_weights = F.softmax(scaled_dot_product, dim=-1)
        
        # 5. 가중치에 맞춰 실제 데이터(V) 가져오기
        out = attention_weights @ V
        
        return out, attention_weights

# --------------------------------------------------------
# 시각화: 어텐션 맵(Attention Map) 들여다보기
# --------------------------------------------------------
# 가상의 오디오 스펙트로그램 특징 맵 준비 
# (배치 1, 시간 10 프레임, 각 프레임당 16차원 특징)
seq_len = 10
embed_dim = 16
dummy_audio_features = torch.randn(1, seq_len, embed_dim)

# 어텐션 모듈 통과
attention_module = SelfAttention(embed_size=embed_dim)
output_features, attn_map = attention_module(dummy_audio_features)

# 어텐션 맵 이미지로 출력 (시간 vs 시간)
attn_matrix = attn_map.squeeze().detach().numpy()

plt.figure(figsize=(6, 5))
plt.imshow(attn_matrix, cmap='viridis')
plt.colorbar(label='Attention Weight (0~1)')
plt.title("Self-Attention Map (Time vs Time)", fontsize=14)
plt.xlabel("Key Time Steps ($K$)", fontsize=12)
plt.ylabel("Query Time Steps ($Q$)", fontsize=12)
plt.show()
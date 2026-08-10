# Chapter 16. 전체 신호의 맥락을 읽다: 트랜스포머(Transformer)와 셀프 어텐션(Self-Attention)

- 15장에서 우리는 팽창 합성곱(Dilated Convolution)을 통해 1D CNN의 시야를 기하급수적으로 넓히는 웨이브넷(WaveNet)의 마법을 보았습니다. 하지만 CNN과 RNN 계열의 모델들은 태생적인 한계를 안고 있습니다. 어쨌든 '가까운 시간대의 데이터'를 우선적으로 보거나 순차적으로 넘겨받아야 한다는 점입니다.

- 만약 교향곡을 분석하는데, 1초에 등장한 '도입부 멜로디'와 3분 뒤에 등장하는 '하이라이트 멜로디'의 완벽한 연관성을 한 번에 찾아내고 싶다면 어떻게 해야 할까요?

- 2017년, 구글이 "Attention Is All You Need"라는 논문으로 발표한 트랜스포머(Transformer)는 이 물리적 거리의 한계를 완전히 박살 냅니다. 자연어 처리(NLP)를 정복한 이 괴물 모델은 이제 오디오와 시계열 신호처리 분야(AST, Informer 등)까지 집어삼키고 있습니다.

- 이번 장에서는 트랜스포머의 심장, 셀프 어텐션(Self-Attention) 메커니즘을 쌩코딩으로 해부해 보겠습니다.

## 1. 4장의 기억: 내적(Dot Product)의 화려한 귀환

- 본격적인 구조에 앞서, 우리가 2부 4장에서 푸리에 변환(FFT)을 배울 때 썼던 무기를 다시 꺼내봅시다.
- 복잡한 스무디(신호) 속에서 특정 주파수(딸기)를 찾기 위해 우리는 무엇을 썼나요? 바로 내적(Dot Product)이었습니다. 두 배열을 내적 해서 값이 크게 나오면 "서로 닮았다(유사도가 높다)"는 것을 의미했죠.

- 셀프 어텐션의 본질도 완벽하게 똑같습니다!
- 차이가 있다면, 신호와 '사인파'를 내적 하는 것이 아니라 '신호 자기 자신(Self)의 다른 시간대 데이터'와 내적을 수행한다는 점입니다. "1초일 때의 내 소리 패턴과 가장 닮아있는 소리가 저 멀리 10초 뒤에 또 있을까?"를 내적으로 싹 다 찔러보는(탐색하는) 무식하지만 확실한 방법입니다.

## 2. 질문(Q), 열쇠(K), 그리고 내용물(V)

- 트랜스포머는 이 내적 기반의 탐색을 아주 우아한 시스템으로 구축했습니다. 바로 데이터베이스 검색 시스템과 똑같은 Q, K, V 메커니즘입니다.

- 신호의 특정 시간대($t=1$) 데이터가 다른 시간대($t=5$) 데이터와 연관이 있는지 알아보기 위해, 신호를 세 가지의 다른 역할로 복제(변환)합니다.

1. Query ($Q$, 질문): "나는 $t=1$의 드럼 비트야. 나와 박자가 맞는 짝꿍 소리를 찾고 싶어!" (내가 찾는 조건)
2. Key ($K$, 열쇠/태그): "나는 $t=5$의 스네어 드럼 소리야!" (나의 정체성)
3. Value ($V$, 내용물): $t=5$에 들어있는 실제 소리의 특징 데이터.

- 어텐션의 동작 원리는 다음과 같습니다.
- $t=1$의 $Q$가 모든 시간대의 $K$들과 내적(Dot Product)을 수행하여 유사도 점수를 매깁니다. 만약 $t=5$의 $K$와 찰떡궁합이라 점수가 높게 나왔다면, $t=5$의 실제 데이터인 $V$를 듬뿍 가져와서 $t=1$의 정보에 섞어줍니다. 반대로 점수가 낮으면(관련 없는 노이즈면) 가져오지 않습니다.

## 3. 수식 해부: 소맥(Softmax)과 내적의 콜라보

- 이 과정을 수식으로 표현하면 딥러닝 역사상 가장 유명한 공식이 됩니다.

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

- $Q K^T$: $Q$와 $K$를 내적(행렬 곱)하여 어텐션 맵(Attention Map, 유사도 점수판)을 만듭니다.
- $\text{softmax}$: 유사도 점수를 0~1 사이의 확률(비율) 값으로 예쁘게 바꿔줍니다. (합치면 100%가 되도록)
- $V$: 최종적으로 비율에 맞춰 $V$(실제 데이터)를 곱해서 가져옵니다.

## 4. 파이토치로 조립하는 Self-Attention (Python 실습)
- 수식만 보면 복잡해 보이지만, 파이토치(PyTorch)의 행렬 곱셈(torch.matmul 또는 @)을 사용하면 단 몇 줄의 코드로 우아하게 구현됩니다.

```python
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
```

(Attention Map)
- 코드를 실행해서 출력된 $10 \times 10$ 크기의 어텐션 맵 이미지를 자세히 살펴보세요.
- Y축은 '현재 내가 있는 시간($Q$)', X축은 '내가 바라보고 있는 시간($K$)'입니다.

- 만약 3행 8열의 색깔이 아주 밝은 노란색(값이 높음)이라면, 이것은 딥러닝 모델이 "3초 구간의 소리를 이해하기 위해, 멀리 떨어진 8초 구간의 소리 정보가 엄청나게 중요하구나!"라고 스스로 깨닫고 그 정보를 끌어다 쓰고 있다는 뜻입니다.

- CNN처럼 바로 옆 데이터부터 한 칸씩 답답하게 이동할 필요 없이, 원하는 데이터가 1분 뒤에 있든 10분 뒤에 있든 한 방의 내적 연산으로 다이렉트 고속도로를 뚫어버리는 것. 이것이 트랜스포머가 신호처리 판을 뒤집은 비결입니다.

## 5. 오디오 스펙트로그램 트랜스포머 (AST)

- 이 아이디어를 10장에서 만든 2차원 '스펙트로그램' 이미지에 적용한 것이 바로 최신 SOTA 모델인 AST(Audio Spectrogram Transformer)입니다.
- 스펙트로그램 이미지를 작은 네모 조각(Patch)들로 잘게 쪼갠 뒤, 위에서 만든 Self-Attention 모듈에 던져줍니다. 그러면 모델은 "오른쪽 끝에 있는 고음역대 조각이, 왼쪽 아래에 있는 저음역대 조각과 밀접한 연관이 있네!"라는 복잡한 주파수-시간 역학 관계를 스스로 학습해 냅니다.

## Summary
- CNN과 RNN은 근처에 있는 데이터를 중심으로 순차적 처리를 하지만, 트랜스포머는 시간적/물리적 거리를 무시하고 데이터를 처리한다.
- 셀프 어텐션(Self-Attention)은 4장에서 배운 내적(Dot Product)을 활용해 데이터($Q$)가 자기 자신($K$)의 어느 부분을 주의 깊게 봐야 할지 점수판(Attention Map)을 만든다.
- 오디오 데이터를 작은 조각(Patch)으로 잘라 어텐션을 적용하면, 기존 CNN을 능가하는 강력한 음성 분류/인식 성능(AST)을 발휘한다.

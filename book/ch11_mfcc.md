# Chapter 11. 인간의 귀를 모방하다: 멜 스케일과 MFCC의 탄생

- 10장에서 우리는 1차원 소리 배열을 2차원 열화상 사진(스펙트로그램)으로 변환하여, 소리를 눈으로 보는 쾌거를 이루었습니다. 하지만 이 스펙트로그램을 그대로 인공지능(AI)에게 먹여주면, AI는 종종 엉뚱한 결과를 냅니다. 왜 그럴까요?

- 기계의 귀(FFT)는 너무나도 '공평(Linear)'하기 때문입니다.
- 기계는 100Hz와 200Hz의 차이(100Hz 차이)나, 10,000Hz와 10,100Hz의 차이(100Hz 차이)를 물리적으로 완전히 똑같은 비중으로 다룹니다. 스펙트로그램의 픽셀 수도 똑같이 할당하죠.

- 하지만 인간의 귀는 불공평(Non-linear)합니다. 우리는 저음역대(목소리의 특징이 몰려있는 곳)에서는 1Hz의 미세한 음정 변화도 귀신같이 잡아내지만, 고음역대(바람 소리, 쇳소리)에서는 1,000Hz가 휙휙 바뀌어도 그 차이를 잘 구분하지 못합니다.

- 이번 장에서는 기계의 너무 정직한 주파수 축을 인간의 귀처럼 구부러뜨리는(Warping) 마법, **멜 필터뱅크(Mel-Filterbank)** 와 음성 인식의 전설적인 특징 추출 기법인 **MFCC** 를 밑바닥부터 구현해 보겠습니다.

## 1. 주파수 축을 구부려라: 멜 스케일(Mel Scale)

- 1937년, 스티븐스(Stevens) 등은 사람들을 모아놓고 소리의 높낮이를 얼마나 잘 구분하는지 실험했습니다. 
- 그 결과, 인간의 귀가 느끼는 주파수 감각을 수학적으로 모델링한 멜 스케일(Mel Scale)이라는 공식을 만들어냅니다. ('멜'은 멜로디에서 따온 이름입니다.)

- 공식은 로그($\log$) 함수를 사용하여 고주파로 갈수록 값을 꾹꾹 눌러 압축하는 형태를 띱니다.

$$M(f) = 2595 \cdot \log_{10}\left(1 + \frac{f}{700}\right)$$

- 이 공식의 핵심은 단순합니다.
- "저주파(0~1000Hz)에서는 주파수(Hz)와 멜(Mel)이 거의 1:1로 비례하지만, 1000Hz가 넘어가는 고주파에서는 주파수가 아무리 커져도 멜 값은 아주 찔끔찔끔 올라간다!"


## 2. 멜 필터뱅크 (Mel-Filterbank): 겹치는 삼각형들
- 그렇다면 이미 만들어진 선형 스펙트로그램을 어떻게 멜 스케일로 바꿀 수 있을까요?
- 여기서 신호처리의 아주 우아한 테크닉인 삼각 필터뱅크(Triangular Filterbank)가 등장합니다.

- 스펙트로그램의 주파수 축(Y축) 위에 여러 개의 '삼각형 모양의 창문(Filter)'을 겹치게 덮어씌운다고 상상해 봅시다.
1. 저주파 영역: 삼각형들의 폭이 아주 좁고 빽빽하게 모여 있습니다. (미세한 차이도 꼼꼼하게 살핀다.)
2. 고주파 영역: 삼각형들의 폭이 펑퍼짐하게 넓고 듬성듬성 퍼져 있습니다. (넓은 대역의 에너지를 하나로 퉁쳐서 합쳐버린다.)

- 이 수십 개의 삼각형(보통 40개~80개를 씁니다)이 스펙트로그램의 한 줄(특정 시간의 주파수 성분)과 내적(Dot Product)을 수행하여, 데이터를 확 압축해 버립니다.

## 3. 밑바닥부터 짜는 멜 필터뱅크 (Python 실습)
- 수식의 두려움을 없애기 위해, `librosa.filters.mel` 함수가 속에서 정확히 어떻게 돌아가는지 `numpy`로 쌩코딩해 봅시다.

```python
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
```

( 그래프 해석 )
- 코드를 실행해 보면 정말로 왼쪽(저주파)에는 뾰족하고 얇은 삼각형들이 빽빽하게 뭉쳐있고, 오른쪽(고주파)으로 갈수록 거대하고 완만한 삼각형들이 듬성듬성 자리 잡고 있는 것을 볼 수 있습니다.

- 이제 이 필터 행렬(`filterbank_matrix`)과 10장에서 만든 선형 스펙트로그램 행렬을 행렬 곱(Matrix Multiplication, `np.dot`) 한 방으로 때려주면, 인간의 귀로 들은 것과 똑같은 '멜 스펙트로그램(Mel-Spectrogram)'이 탄생합니다!

## 4. 딥러닝 이전 시대의 제왕: MFCC
- 멜 스펙트로그램에 로그(Log)를 씌운 것만으로도 현대의 CNN 기반 딥러닝 모델들은 아주 학습을 잘합니다. (실제로 요즘은 MFCC보다 Log Mel-Spectrogram을 더 많이 씁니다.)

- 하지만 딥러닝이 발달하기 전, 과거의 머신러닝 알고리즘들(GMM, HMM 등)은 멜 스펙트로그램조차 너무 복잡해서 소화하지 못했습니다. 왜냐하면 겹치는 삼각형 필터들을 썼기 때문에, 1번 필터의 결과와 2번 필터의 결과가 서로 너무 비슷해서(상관관계가 높아서) 모델이 헷갈려 했기 때문입니다.

- 이 상관관계를 완전히 박살 내고, 데이터의 크기를 한 번 더 극단적으로 압축하기 위해 수학자들은 이산 코사인 변환(DCT, Discrete Cosine Transform)이라는 기술을 적용했습니다.

- 로그 멜 스펙트로그램 행렬에 이 DCT 연산을 곱하여 나온 최종 결과물을 MFCC(Mel-Frequency Cepstral Coefficients)라고 부릅니다.
- MFCC는 음성의 핵심적인 특징(사람의 성대와 구강 구조 모양)만 남기고, 의미 없는 고주파 노이즈와 음의 높낮이(Pitch) 정보는 싹 날려버리는 극한의 압축 기술입니다. 덕분에 수십 년간 음성 인식 분야의 '절대 반지'로 군림할 수 있었습니다.

## Summary
- 기계의 귀(FFT)는 선형적이지만, 인간의 귀는 고음의 미세한 차이를 잘 듣지 못하는 비선형적(Mel Scale) 특성을 가진다.
- 멜 필터뱅크는 저주파는 촘촘히, 고주파는 넓게 덮는 겹치는 삼각형 행렬로, 선형 스펙트로그램과 곱해져 데이터를 인간의 청각과 비슷하게 압축한다.
- 여기에 DCT(이산 코사인 변환)를 한 번 더 거쳐 정보를 극도로 압축하고 특징만 남긴 것이 음성 인식의 근본 특징인 MFCC다.

- 이제 우리는 세상의 물리적인 소리를 파이썬 배열로 가져와(1부), 주파수 세계를 해부하고(2부), 정밀하게 다듬은 뒤(3부), 마침내 인공지능이 씹어먹기 가장 완벽한 형태인 2차원 '멜 스펙트로그램 / MFCC'로 가공하는(4부) 기나긴 신호처리 파이프라인을 바닥부터 완성했습니다!
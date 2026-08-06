from scipy.io import wavfile

# 실제 음성 파일 경로 (독자 제공용 파일)
file_path = "./example/example.wav" 

# wav 파일 읽기
# 반환값 1: sample_rate (이 파일이 1초에 몇 번 샘플링되었는가?)
# 반환값 2: audio_array (실제 소리 데이터가 담긴 1차원 넘파이 배열!)
sample_rate, audio_array = wavfile.read(file_path)

print(f"샘플링 레이트 (fs): {sample_rate} Hz")
print(f"배열의 크기: {audio_array.shape}")
print(f"이 오디오의 실제 길이: {len(audio_array) / sample_rate:.2f} 초")

# 앞부분 10개의 데이터만 슬쩍 확인해보기
print(f"데이터 샘플 (처음 10개): {audio_array[:10]}")
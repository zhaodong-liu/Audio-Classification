import librosa
import soundfile
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import copy

# path = "train_mp3s/"

# filename = path + "0.mp3"

# y, sr = librosa.load(filename, sr=None)
# D = np.abs(librosa.stft(y))**2
# S = librosa.feature.melspectrogram(y=y, sr=sr)
# print(S.shape)
#     # Passing through arguments to the Mel filters
# S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
# print(S.shape)
# print(type(S))
# # Save the value of S in a CSV file
# with open('output.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows(S)

# S = torch.tensor(S)
# print(S.shape)
# S = S.flatten()
# print(S.shape)
# plt.figure()
# plt.plot(S)
# plt.show()
# S = S.numpy()
# print(type(S))
# with open('output.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows(S)

# waveform,sample_rate = torchaudio.load(filename)
# print("Shape of waveform:{}".format(waveform.size()))
# print("sample rate of waveform:{}".format(sample_rate)) # 44100
# plt.figure()
# plt.plot(waveform.t().numpy())
# plt.show()

# specgram = torchaudio.transforms.MelSpectrogram()(waveform) # tensor Size([1, 128, 662])
# print("Shape of spectrogram:{}".format(specgram.size()))
# print(type(specgram))

# plt.figure()
# plt.plot(specgram[0].t().numpy())
# plt.show()




sr = 16000
n_fft = 1024
frame_shift = 0.0125
frame_length = 0.05
hop_length = int(sr * frame_shift) # 200
print(hop_length)
win_length = int(sr * frame_length) # 800
print(win_length)
n_mels = 80
power = 1.2
n_iter = 100
preemphasis = 0.97
max_db = 100
ref_db = 20
top_db = 15

path = "train_mp3s/"

# def get_spectrograms(path,sr = 16000,
#         n_fft = 1024,
#         frame_shift = 0.0125,
#         frame_length = 0.05,
#         hop_length = 200,
#         win_length = 800,
#         n_mels = 80,
#         power = 1.2,
#         n_iter = 100,
#         preemphasis = 0.97,
#         max_db = 100,
#         ref_db = 20,
#         top_db = 15):

def get_spectrograms(path):
    
    hop_length = int(sr * frame_shift)
    win_length = int(sr * frame_length)

    y, sr = librosa.load(path, sr=16000)
    y = np.append(y[0], y[1:] - preemphasis * y[:-1])
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)  #stft
    S = np.abs(D) ** power  # magnitude spectrogram
    mel_basis = librosa.filters.mel(sr, n_fft, n_mels) # (n_mels, 1+n_fft//2)
    S = np.dot(mel_basis, S) # (n_mels, t)
    S = 20 * np.log10(np.maximum(1e-5, S))    # dB
    S = np.clip((S - ref_db + max_db) / max_db, 1e-8, 1)
    return S


S = get_spectrograms(path + "0.mp3")
print(S.shape)
import librosa
from matplotlib import pyplot as plt

path = "train_mp3s/"
for i in range(11):
    y, fs = librosa.load(path + str(i) + ".mp3")
    fig,axs = plt.subplots(nrows=1,ncols=1)
    print(y)
    print(fs)
    print(len(y))
    print(len(y)/fs)
    print("\n")
    librosa.display.waveshow(y, sr=fs, ax = axs, color = "blue")

    # plt.show()
    plt.savefig(path+"waveform" + str(i) + ".png")

    X = librosa.stft(y)
    Xdb = librosa.amplitude_to_db(abs(X))
    fig,axs = plt.subplots(nrows=1,ncols=1)
    librosa.display.specshow(Xdb,sr=fs, x_axis='time', y_axis='hz', ax = axs)
    plt.savefig(path+"spectral" + str(i) + ".png")




# y, fs = librosa.load(path + '3.mp3',sr = 16000)
# sf.write(path + '3www.wav',y,fs)
# # print(len(y))
# # yt, index = librosa.effects.trim(y,top_db=30)
# # print(index)
# # fig, axs = plt.subplots(nrows=2,ncols=1)
# # librosa.display.waveshow(y, sr=fs, ax=axs[0])
# # librosa.display.waveshow(yt, sr=fs, ax=axs[1])
# # plt.show()
# # sf.write('test_trim.mp3',yt,fs)

# plt.figure(figsize = (14,5))
# librosa.display.waveshow(y,sr = fs, color = "blue")
# plt.show()
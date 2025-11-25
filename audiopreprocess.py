import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.ticker as ticker
 

path = "train_separate/"
 
def mel_filter_banks(idx, path):
    y, sr = librosa.load(path + "vocal"+ str(idx) + '.mp3', sr=16000)
    
    tmax, tmin = 3, 0
    t = np.linspace(tmin, tmax, (tmax - tmin) * sr)
    
    alpha = 0.97
    emphasized_y = np.append(y[tmin * sr], y[tmin * sr + 1:tmax * sr] - alpha * y[tmin * sr:tmax * sr - 1])
    n = int((tmax - tmin) * sr)  # n = total samples
    frame_size, frame_stride = 0.025, 0.01
    frame_length, frame_step = int(round(sr*frame_size)), int(round(sr*frame_stride))
    # print(frame_length, frame_step)
    signal_length = (tmax-tmin)*sr
    # round up
    frame_num = int(np.ceil((signal_length-frame_length)/frame_step))+1
 
    # add padding to make sure that all frames have equal number of samples without truncating any samples from the original signal
    pad_frame = (frame_num-1)*frame_step+frame_length-signal_length
    pad_y = np.append(emphasized_y, np.zeros(pad_frame))
    signal_len = signal_length+pad_frame
    
    indices = np.tile(np.arange(0, frame_length), (frame_num, 1)) + np.tile(
        np.arange(0, frame_num * frame_step, frame_step), (frame_length, 1)).T

    # every row of frames is the sample value of each frame
    frames = pad_y[indices]

    # add hamming window to each frame
    frames *= np.hamming(frame_length)
    
    # frame_length=1102, 1024 is OK
    NFFT =1024
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    pow_frames = mag_frames ** 2 / NFFT
    
    # define mel filter banks
    mel_N = 40
    mel_low, mel_high = 0, (2595 * np.log10(1 + (sr / 2) / 700))
    mel_freq = np.linspace(mel_low, mel_high, mel_N + 2)
    hz_freq = (700 * (10 ** (mel_freq / 2595) - 1))
    # convert Hz to fft bin number
    bins = np.floor((NFFT) * hz_freq / sr)
    # save data of mel filter banks
    fbank = np.zeros((mel_N, int(NFFT / 2 + 1)))
    for m in range(1, mel_N + 1):
        f_m_minus = int(bins[m - 1])  # left
        f_m = int(bins[m])  # center
        f_m_plus = int(bins[m + 1])  # right
    
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bins[m - 1]) / (bins[m] - bins[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bins[m + 1] - k) / (bins[m + 1] - bins[m])
    filter_banks = np.matmul(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # np.finfo(float) is the smallest positive float
    filter_banks = 20 * np.log10(filter_banks)
    # filter_banks -= np.mean(filter_banks,axis=1).reshape(-1,1)
    # print(filter_banks.shape)          # (299, 40)
    # print(type(filter_banks))
    # filter_banks = filter_banks.flatten()
    # print(filter_banks.shape)

    np.savez(path+str(idx)+'feature.npz', filter_banks)



if __name__ == "__main__":
    n = 0
    while n < 11886:
        for j in range(n, n+100):
            if j == 11886:
                break
            mel_filter_banks(j, path)
            print(j)
        n += 100

    # mel_filter_banks(2446, path)

    # for i in range(8660, 8680):
    #     mel_filter_banks(i, path)
    #     # librosa_mel_filter(i, path)
    #     print(i)



    print('done')
        

    
    
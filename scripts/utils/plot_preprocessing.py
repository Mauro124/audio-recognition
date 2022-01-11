import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import logfbank, mfcc
import librosa
import librosa.display
from configuration import Config

config = Config(mode='conv')
sound_dir = config.sound_split_dir_test
csv_dir = config.csv_test
column = 'fname'


def plot_signals(signals):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(20, 5))
    fig.suptitle('Time Series', size=16)

    i = 0
    for x in range(2):
        for y in range(5):
            axes[x, y].set_title(list(signals.keys())[i])
            axes[x, y].plot(list(signals.values())[i])
            axes[x, y].get_xaxis().set_visible(False)
            axes[x, y].get_yaxis().set_visible(False)
            i += 1


# La FFT tiene 2 partes, la magnitud (Y) y la componente en frecuencia (freq).
def calc_fft(y, rate):
    n = len(y)
    freq = np.fft.rfftfreq(n, d=1/rate)
    Y = abs(np.fft.rfft(y)/n)
    return (Y, freq)


def plot_fft(fft):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(20, 5))
    fig.suptitle('Fourier Transforms', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            data = list(fft.values())[i]
            Y, freq = data[0], data[1]
            axes[x, y].set_title(list(fft.keys())[i])
            axes[x, y].plot(freq, Y)
            axes[x, y].get_xaxis().set_visible(False)
            axes[x, y].get_yaxis().set_visible(False)
            i += 1


def plot_fbank(fbank):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(20, 5))
    fig.suptitle('Filter Bank Coefficients', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x, y].set_title(list(fbank.keys())[i])
            axes[x, y].imshow(list(fbank.values())[i],
                              cmap='hot', interpolation='nearest')
            axes[x, y].get_xaxis().set_visible(False)
            axes[x, y].get_yaxis().set_visible(False)

            i += 1


def plot_mfccs(mfccs):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(20, 5))
    fig.suptitle('Mel Frequency Cepstrum Coefficients', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x, y].set_title(list(mfccs.keys())[i])
            axes[x, y].imshow(list(mfccs.values())[i],
                              cmap='hot', interpolation='nearest')
            axes[x, y].get_xaxis().set_visible(False)
            axes[x, y].get_yaxis().set_visible(False)
            i += 1


def read_csv_audio_test():
    df = pd.read_csv(config.csv_test, dtype=str)
    print(df)
    df.set_index(column, inplace=True)

    for f in df.index:
        rate, signal = wavfile.read(config.sound_dir_test + f)
        df.at[f, 'length'] = signal.shape[0]/rate

    classes = list(np.unique(df.label))
    class_dist = df.groupby(['label'])['length'].mean()
    return classes, class_dist, df


def preprocessing_audio_example():

    print('-------------------')
    print('PROCESANDO IMAGENES')
    print('-------------------')

    classes, class_dist, df = read_csv_audio_test()

    signals = {}
    fft = {}
    fbank = {}
    mfccs = {}

    for c in classes:
        print('PROCESANDO: ' + c)
        wav_file = df[df.label == c].iloc[:3]
        signal, rate = librosa.load(
            config.sound_dir_test + wav_file.index[0], sr=44100)
        signals[c] = signal

        fft[c] = calc_fft(signal, rate)
        bank = logfbank(signal[:rate], rate, nfilt=config.nfilt, nfft=1103).T
        fbank[c] = bank

        mel = mfcc(signal[:rate], rate, nfilt=config.nfilt,
                   nfft=1103, numcep=config.numcep).T
        mfccs[c] = mel

    plot_signals(signals)
    plt.show()

    plot_fft(fft)
    plt.show()

    plot_fbank(fbank)
    plt.show()

    plot_mfccs(mfccs)
    plt.show()


preprocessing_audio_example()

from configuration import Config
import os
from glob import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy.io import wavfile
import csv

config = Config('conv')


def read_csv_audio(path):
    print('------------------')
    print('READING CSV AUDIOS')
    print('------------------')
    df = pd.read_csv(config.csv_path, dtype=str)
    df.set_index('dir_fname', inplace=True)
    for f in df.index:
        rate, signal = wavfile.read(path + f)
        df.at[f, 'length'] = signal.shape[0]/rate

    classes = list(np.unique(df.label))
    class_dist = df.groupby(['label'])['length'].mean()
    return classes, class_dist, df

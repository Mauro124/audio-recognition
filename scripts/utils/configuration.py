import os


class Config:
    # DIRS
    path = os.path.abspath(os.getcwd())
    print('DIRECTORIO PRINCIPAL: ' + path)

    assets = path + '\\assets\\'
    wavfiles = assets + '\\wavfiles\\'
    spectogram_test_dir = assets + '\\test\\spectograms\\'
    spectogram_train_dir = assets + '\\spectograms\\'
    sound_test_clean_dir = assets + '\\test\\clean\\'
    sound_dir_test = wavfiles + '\\test\\'
    sound_split_dir_test = assets + '\\test\\split_audio\\'
    csv_test = assets + 'CSVs\\test.csv'
    input_train_dir = path + '\\inputs\\'
    models_dir = path + '\\models\\'

    def __init__(self, mode='conv', nfilt=26, nfeat=13, nfft=1103, rate=16000, threshold=0.5, numcep=13):
        self.mode = mode
        self.numcep = numcep
        self.nfilt = nfilt
        self.threshold = threshold
        self.nfft = nfft
        self.nfeat = nfeat
        self.rate = rate
        self.step = int(rate/20)
        self.model_path = os.path.join('models', mode + '.model')
        self.p_path = os.path.join('pickles', mode + '.p')

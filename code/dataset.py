from pathlib import Path
import math

import keras
import numpy as np


class DataGenerator(keras.utils.Sequence):
    def __init__(self, basedir, batch_size):
        paths = [f for f in basedir.iterdir() if f.is_file()]
        paths = list(sorted(paths, key=lambda x : int(str(x.stem).split('_')[-1])))
        self.files = [f.as_posix() for f in paths]
        self.batch_size = batch_size

    def load(self, path):
        loaded = np.load(path)
        x = loaded['x']/ 255
        y = loaded['y']/ 255
        return x, y

    def __len__(self):
        return math.ceil(len(self.files) / self.batch_size)

    def __getitem__(self, idx):
        low = idx * self.batch_size
        high = min(low + self.batch_size, len(self.files))
        batch = self.files[low:high]
        X, Y = [],[]
        for f in batch:
            x,y = self.load(f)
            X.append(x)
            Y.append(y)

        return np.concatenate(X), np.concatenate(Y)


if __name__ == '__main__':
    datagen = DataGenerator((Path('..') / 'data' / 'Part_01' / 'crop_1'), 16)
    _x, _y = datagen[0]
    _x.shape, _y.shape
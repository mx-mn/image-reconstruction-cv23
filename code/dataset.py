import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from pathlib import Path
import torch

def load(path: Path, normalize=True):
    # load the mini dataset
    loaded = np.load(path)
    x, y, labels = loaded['x'], loaded['y'], loaded['labels']

    if normalize:
        y = y / 255
        x = x / 255

    return x, y, labels


class AOSDataset(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.files[idx]
        loaded = np.load(path)
        x = torch.from_numpy(loaded['x'])
        y = torch.from_numpy(loaded['y'])
        return x, y


if __name__ == '__main__':
    pass
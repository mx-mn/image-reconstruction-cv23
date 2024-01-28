'''All code is in this file. Because when we want to run this on colab, just copy the complete file into a cell and run.'''

import os
os.environ["KERAS_BACKEND"] = "torch"


import matplotlib.pyplot as plt
from pathlib import Path
import math

import keras
from keras.layers import Input, Conv2D, Conv2DTranspose, Add, Activation
from keras.models import Model
import numpy as np
import random
from collections import defaultdict
from tqdm import tqdm

map_label_to_name = ['no_person', 'idle','sitting', 'laying']

class DataGenerator(keras.utils.Sequence):
    def __init__(
        self,
        basedir: Path,
        batch_size: int = None,
        included_poses: list = None,
        included_trees: list = None,
        shuffle=False,
        only_use_n: int = None,
        random_rotation: bool = False,
        random_flip: bool = False,
    ):
        if not basedir.exists():
            ValueError('Datafolder does not exist. Add it to your drive and try again. Maybe restart the runtime.')

        self.basedir = basedir
        self.batch_size = batch_size
        self.included_poses = [map_label_to_name.index(pose) for pose in included_poses] if included_poses is not None else None
        self.included_trees  = included_trees
        self.filenames = self.__filter(shuffle, only_use_n)
        self.random_flip = random_flip
        self.random_rotation = random_rotation

    def __filter(self, shuffle, only_use_n):

        files = []
        self.pose_distribution = defaultdict(int)
        self.trees_distribution = defaultdict(int)
        self.pose_distribution_filtered = defaultdict(int)
        self.trees_distribution_filtered = defaultdict(int)

        unfiltered = list(self.basedir.iterdir())

        if shuffle:
            random.shuffle(unfiltered)

        total = len(unfiltered)
        if only_use_n is not None:
            total = only_use_n

        for path in tqdm(unfiltered, total=total):

            loaded = np.load(path)
            pose, trees = loaded['pose'], loaded['trees']

            self.pose_distribution[pose.item()] += 1
            self.trees_distribution[trees.item()] += 1

            fname = path.name
            if self.included_poses is not None and pose not in self.included_poses:
                continue

            if self.included_trees is not None and trees not in self.included_trees:
                continue

            files.append(fname)
            self.pose_distribution_filtered[pose.item()] += 1
            self.trees_distribution_filtered[trees.item()] += 1

            if only_use_n is not None and len(files) == only_use_n:
                break

        return files

    def load(self, path):
        loaded = np.load(path)
        x = loaded['x'] / 255
        y = loaded['y'] / 255
        return x, y
    
    def __len__(self):
        if self.batch_size is None:
            return len(self.filenames)

        return math.ceil(len(self.filenames) / self.batch_size)

    def __getitem__(self, idx):

        if self.batch_size is None:
            batch = self.filenames
        else:
            low = idx * self.batch_size
            high = min(low + self.batch_size, len(self.filenames))
            batch = self.filenames[low:high]

        X, Y = [],[]
        for fname in batch:
            x,y = self.load(self.basedir / fname)

            flip = self.random_flip and bool(random.getrandbits(1))

            x = np.fliplr(x) if flip else x
            y = np.fliplr(y) if flip else y
            X.append(x)
            Y.append(y)

        return np.stack(X), np.stack(Y)

    def print_info(self):
        print()
        shape = self.load(self.basedir / self.filenames[0])[0].shape
        print(f'{len(self.filenames)} samples with shape : {shape}')

        print(f'Pose distribution total')
        ("{:<15} {:<15}".format('pose', 'number of samples'))
        for key, value in self.pose_distribution.items():
            print("{:<15} {:<15}".format(map_label_to_name[key], value))
        print()
        print(f'Pose distribution filtered')
        ("{:<15} {:<15}".format('pose', 'number of samples'))
        for key, value in self.pose_distribution_filtered.items():
            print("{:<15} {:<15}".format(map_label_to_name[key], value))

        print()
        print(f'Trees distribution total')
        print("{:<15} {:<15}".format('num trees per ha', 'number of samples'))

        for key, value in self.trees_distribution.items():
            print("{:<15} {:<15}".format(key, value))

        print()
        print(f'Trees distribution filtered')
        print("{:<15} {:<15}".format('num trees per ha', 'number of samples'))

        for key, value in self.trees_distribution_filtered.items():
            print("{:<15} {:<15}".format(key, value))


def select_random_samples(*arrs, num=100):
    indices = np.random.choice(arrs[0].shape[0], size=num, replace=False)
    return [arr[indices] for arr in arrs]


# TRAINING WITH KERAS
class PredictionCallback(keras.callbacks.Callback):
    def __init__(self, interval, x_val, y_val):
        super(PredictionCallback, self).__init__()
        self.interval = interval
        self.x_val = x_val
        self.y_val = y_val

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            preds = self.model.predict(self.x_val).squeeze()
            plot_image_grid(np.concatenate([self.y_val, preds]), preds.shape[0], 2)

def plot_history(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('MSE Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

def plot_image_grid(images_array, grid_width=10, grid_height=10):

    if images_array.shape[0] != grid_width * grid_height:
        raise ValueError("The number of images does not match the grid size.")
    
    fig, axes = plt.subplots(grid_height, grid_width, figsize=(grid_width, grid_height))
    
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(images_array[i], cmap='gray', interpolation='none')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


# MODEL
def encoder(x, num_features, num_layers, residual_every=2):
    x = Conv2D(num_features, kernel_size=3, strides=2, padding='same', activation='relu')(x)

    # Save the output of conv layers at even indices
    residuals = []

    # Encoder
    for i in range(num_layers - 1):
        x = Conv2D(num_features, kernel_size=3, padding='same', activation='relu')(x)
        if (i + 1) % residual_every == 0:
            residuals.append(x)

    return x, residuals

def decoder(x, num_features, num_layers, residuals, residual_every=2):

    # Decoder
    for i in range(num_layers - 1):
        x = Conv2DTranspose(num_features, kernel_size=3, padding='same')(x)

        if (i + 1 + num_layers) % residual_every == 0 and residuals:
            res = residuals.pop()
            x = Add()([x, res])
            
        x = Activation('relu')(x)

    if residuals: raise ValueError('There are unused residual connections')

    # create 1-channel output
    x = Conv2DTranspose(1, kernel_size=3, strides=2, padding='same')(x)

    return x

def REDNet(num_layers, num_features, channel_size):
    '''Model definition with keras functional layers api'''

    inputs = Input(shape=(None, None, channel_size))

    x, residuals = encoder(inputs, num_features, num_layers)

    x = decoder(x, num_features, num_layers, residuals)

    # Add input residual, needed to do 1x1 conv to adapt channels
    residual = Conv2DTranspose(1, kernel_size=1, padding='same')(inputs)
    outputs = Add()([x, residual])
    outputs = Activation('relu')(outputs)

    # Create model
    model = Model(inputs=inputs, outputs=outputs, name=f'REDNet{num_layers*2}')
    return model


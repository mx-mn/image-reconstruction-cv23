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

class DataGenerator(keras.utils.Sequence):
    '''
    This can be used in keras.Model.fit method. It loads .npz files from disk, keeps RAM usage low.
    Locally sometimes there were errors. couldnt import keras.utils.Sequence, but on colab it works.

    TODO:
    - make it support multiple directories from where .npz files are accumulated. 
    '''
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


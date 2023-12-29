import matplotlib.pyplot as plt

import os
os.environ["KERAS_BACKEND"] = "torch"

import keras
import numpy as np

def plot_image_grid(images_array, grid_width=10, grid_height=10):

    if images_array.shape[0] != grid_width * grid_height:
        raise ValueError("The number of images does not match the grid size.")
    
    fig, axes = plt.subplots(grid_height, grid_width, figsize=(grid_width, grid_height))
    
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(images_array[i], cmap='gray', interpolation='none')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_history(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('MSE Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

def select_random_samples(*arrs, num=100):
    indices = np.random.choice(arrs[0].shape[0], size=num, replace=False)
    return [arr[indices] for arr in arrs]

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

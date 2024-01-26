import keras
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.layers import Input, Conv2D, Conv2DTranspose, Add, Activation
from keras.models import Model
import numpy as np
import math
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path
import random

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

def main():
    root =  Path(__file__).parent.resolve() 
    train_set_path = root / 'demo'
    val_set_path = root / 'demo'
    checkpoint_dir = root / 'checkpoints'

    validation_data = DataGenerator(
        val_set_path, 
        batch_size=1,
        #only_use_n=1,
    )

    train_data = DataGenerator(
        train_set_path,
        batch_size=128,
        included_poses=['idle','sitting', 'laying'],
        shuffle=True,
        random_flip=True,
        #only_use_n=1,
        #batch_size=1
    )

    # compile the model
    model = REDNet(
        num_layers=11,
        num_features=64,
        channel_size=6
    )

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=1000,
        decay_rate=0.9,
        staircase=True
    )
    opt = keras.optimizers.Adam(
        learning_rate=lr_schedule
    )
    loss = keras.losses.MeanSquaredError( reduction="sum_over_batch_size")
    model.compile(loss=loss,optimizer=opt)
    callbacks = [
        ModelCheckpoint((checkpoint_dir / 'ep{epoch:02d}_loss{val_loss:.4f}.keras').as_posix(), save_best_only=True),
        CSVLogger(checkpoint_dir / 'logs.csv', append=True),
    ]

    # Train 40 Epochs with Lr Schedule
    model.fit(
        train_data,
        validation_data=validation_data,
        epochs=40,
        callbacks=callbacks,
        shuffle=True,
        verbose=1
    )

    # Train 20 epochs with fixed Learning Rate
    opt = keras.optimizers.Adam(
        learning_rate=0.0001
    )
    loss = keras.losses.MeanSquaredError( reduction="sum_over_batch_size")
    model.compile(loss=loss,optimizer=opt)
    model.fit(
        train_data,
        validation_data=validation_data,
        epochs=60,
        callbacks=callbacks,
        shuffle=True,
        verbose=1,
        initial_epoch=40,
    )

    # Train 40 epochs more, but with Mean Absolute Error
    opt = keras.optimizers.Adam(
        learning_rate=0.0001
    )
    loss = keras.losses.MeanAbsoluteError(reduction="sum_over_batch_size")
    model.compile(loss=loss,optimizer=opt)
    model.fit(
        train_data,
        validation_data=validation_data,
        epochs=100,
        callbacks=callbacks,
        shuffle=True,
        verbose=1,
        initial_epoch=60
    )


    # Train one last time with lower learning rate
    opt = keras.optimizers.Adam(
        learning_rate=0.00001
    )
    loss = keras.losses.MeanAbsoluteError(reduction="sum_over_batch_size")
    model.compile(loss=loss,optimizer=opt)
    train_data.batch_size=512
    model.fit(
        train_data,
        validation_data=validation_data,
        epochs=113,
        callbacks=callbacks,
        shuffle=True,
        verbose=1,
        initial_epoch=100
    )

if __name__ == '__main__':
    main()
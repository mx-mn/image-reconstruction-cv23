# enter the models here
import os
os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras.layers import Input, Conv2D, Conv2DTranspose, Add, Activation
from keras.models import Model

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


# Example usage
if __name__ == '__main__':

    # Create the REDNet10 model
    model = REDNet(num_layers=5, num_features=64, channel_size=3)

    # rednet10.summary()
    keras.utils.plot_model(
        model, 
        to_file=f"{model.name}.png",
        show_shapes=True, 
        show_trainable=True, 
        show_layer_activations=True,
        rankdir='LR'
        )
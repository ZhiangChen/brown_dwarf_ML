#!/usr/bin/env python
"""
model.py
Zhiang Chen
Nov 24, 2019
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import os

class FCL(object):
    def __init__(self, input_shape, layers, activation_function='relu'):
        """
        :param input_shape: (int)
        :param layers: (list) e.g. [hidden1, hidden2, ..., output]
        :param activation_function:
        """
        self.input_shape = input_shape
        self.layers = layers
        self.activation_fuction = activation_function

        self.model = self._build_model()
        self.model.summary()

    def _build_model(self):
        layers = list()
        layers.append(tf.keras.layers.Dense(self.layers[0], activation=self.activation_fuction, input_shape=[self.input_shape]))
        if len(self.layers) >= 3:
            for layer in self.layers[1:-1]:
                layers.append(tf.keras.layers.Dense(layer, activation=self.activation_fuction))
        layers.append(tf.keras.layers.Dense(self.layers[-1]))

        model = tf.keras.Sequential(layers)
        optimizer = tf.keras.optimizers.RMSprop(0.0001)
        model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
        return model

    def train(self, dataset, epochs, verbose=0, save_model=False, validation_data=None):
        """

        :param dataset:
        :param epochs: In each epoch, all training dataset will be trained once.
        :param verbose: 0 = silent, 1 = progress bar, 2 = one line per epoch
        :return:
        """
        # callbacks
        """
        class PrintDot(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs):
                if epoch % 100 == 0: print('')
                print('.', end='')
        """
        #callbacks.append(PrintDot)
        #early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        callbacks = None
        if save_model:
            checkpoint_path = "training/cp-{epoch:04d}.ckpt"
            checkpoint_dir = os.path.dirname(checkpoint_path)

            # Create a callback that saves the model's weights
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                             save_weights_only=True,
                                                             verbose=2)
            callbacks = [cp_callback]

        history = self.model.fit(dataset, epochs=epochs, verbose=verbose, callbacks=callbacks, validation_data=validation_data)
        return history

    def load_weights(self, ckpt_path):
        self.model.load_weights(ckpt_path)

    def evaluate(self, dataset):
        """
        :param dataset: TF data
        :return:
        """
        return self.model.evaluate(dataset)

    def predict(self, dataset):
        """
        :param dataset: TF data; ndarray
        :return:
        """
        return self.model.predict(dataset)

    def plot_history(self, history, y_mae=0.15, y_mse=0.02):
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch

        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Abs Error [MPG]')
        plt.plot(hist['epoch'], hist['mae'], label='Train Error')
        plt.plot(hist['epoch'], hist['val_mae'], label='Val Error')
        plt.ylim([0, y_mae])
        plt.legend()

        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Square Error [$MPG^2$]')
        plt.plot(hist['epoch'], hist['mse'], label='Train Error')
        plt.plot(hist['epoch'], hist['val_mse'], label='Val Error')
        plt.ylim([0, y_mse])
        plt.legend()
        plt.show()


class Conv1DResNet(object):
    def __init__(self, input_shape):
        self.input_shape = input_shape


    def _build_model(self):
        pass

if __name__ == "__main__":
    inputs = 200
    layers = [200, 20, 2]
    nn = FCL(inputs, layers)


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mae'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'], label='Val Error')
    plt.ylim([0, 0.5])
    plt.legend()
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mse'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'], label='Val Error')
    plt.ylim([0, 0.5])
    plt.legend()
    plt.show()
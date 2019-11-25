#!/usr/bin/env python
"""
model.py
Zhiang Chen
Nov 24, 2019
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from data import *

class FCN(object):
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

    def train(self, data, labels, validation_slipt, epochs):
        # callbacks
        class PrintDot(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs):
                if epoch % 100 == 0: print('')
                print('.', end='')

        #early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        history = self.model.fit(data, labels, epochs=epochs, validation_slipt=validation_slipt, verbose=0, callbacks=[PrintDot()])
        return history

    def plot_history(self, history):
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch

        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Abs Error [MPG]')
        plt.plot(hist['epoch'], hist['mae'], label='Train Error')
        plt.plot(hist['epoch'], hist['val_mae'], label='Val Error')
        plt.ylim([0, 5])
        plt.legend()

        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Square Error [$MPG^2$]')
        plt.plot(hist['epoch'], hist['mse'], label='Train Error')
        plt.plot(hist['epoch'], hist['val_mse'], label='Val Error')
        plt.ylim([0, 20])
        plt.legend()
        plt.show()
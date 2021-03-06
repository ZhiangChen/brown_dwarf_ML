#!/usr/bin/env python
"""
train.py
Zhiang Chen
Nov 24, 2019
"""

from data import *
from model import *

BD = browndwarf()
BD.loadPickles('20_BD_data.pickle', 'truncated_synthetic_data.pickle')
BD.prepareDatasets(data_ratio=.6, synthetic_data_ratio=0.8)
train_flux, train_labels = BD.getTrain(synthetic=True)
train_flux, train_labels, valid_flux, valid_labels = BD.splitData(train_flux, train_labels)
test_flux, test_labels = BD.getTest(synthetic=True)

train_dataset = tf.data.Dataset.from_tensor_slices((train_flux, train_labels))
valid_dataset = tf.data.Dataset.from_tensor_slices((valid_flux, valid_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_flux, test_labels))
BATCH_SIZE = 1
train_dataset = train_dataset.batch(BATCH_SIZE)
valid_dataset = valid_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

input_shape = train_flux[0].shape[0]
layers = [100, 10, 2]

nn = FCL(input_shape, layers)
history = nn.train(train_dataset, epochs=400, verbose=2, validation_data=valid_dataset, save_model=True)
nn.plot_history(history)



"""
from data import *
from model import *

BD = browndwarf()
BD.loadPickles('52_BD_data.pickle', 'truncated_synthetic_data.pickle')
BD.prepareDatasets(data_ratio=.7, synthetic_data_ratio=0.8)
train_flux, train_labels = BD.getTrain(synthetic=False)
train_flux, train_labels, valid_flux, valid_labels = BD.splitData(train_flux, train_labels)
test_flux, test_labels = BD.getTest(synthetic=False)

train_dataset = tf.data.Dataset.from_tensor_slices((train_flux, train_labels))
valid_dataset = tf.data.Dataset.from_tensor_slices((valid_flux, valid_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_flux, test_labels))
BATCH_SIZE = 1
train_dataset = train_dataset.batch(BATCH_SIZE)
valid_dataset = valid_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

input_shape = train_flux[0].shape[0]
layers = [100, 10, 2]
# leaky_relu
lrelu = lambda x: tf.nn.leaky_relu(x, alpha=0.1)
nn = FCL(input_shape, layers, activation_function=lrelu)
#nn.load_weights("./FCN_52_training/cp-0800.ckpt")
nn.evaluate(test_dataset)
history = nn.train(train_dataset, epochs=200, verbose=2, validation_data=valid_dataset, save_model=True)
nn.plot_history(history,0.5,0.5)
"""

#nn.load_weights('training/cp-0100.ckpt')
#print(nn.evaluate(test_dataset))
#print(nn.predict(test_flux))
#print(test_labels)

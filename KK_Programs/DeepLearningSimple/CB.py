import tensorflow as tf
import time

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import numpy as np


class MyCallback(keras.callbacks.Callback):
    def __init__(self):
        self.Val_losses = []
    # def on_epoch_end(self, epoch, logs=None):
    #     keys = list(logs.keys())
    #     print("End epoch {} of training; got log keys: {}".format(epoch, keys))

    def on_epoch_end(self,epoch, logs=None):
        self.Val_losses.append(logs["val_loss"])
    # def on_batch_end(self,logs=None):
    #     self.Val.append(logs["loss"])


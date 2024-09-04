import tensorflow 
from tensorflow import keras


class MyCallback(keras.callbacks.Callback):
    def __init__(self):
        self.Val_losses = []
        self.Val = []
    def on_epoch_end(self,epoch,logs=None):
        self.Val_losses.append(logs["val_loss"])
    # def on_batch_end(self,logs=None):
    #     self.Val.append(logs["loss"])


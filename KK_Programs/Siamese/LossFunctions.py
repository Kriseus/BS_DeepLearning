import numpy as np
# from abc import abstractmethod, ABC
import tensorflow as tf
from tensorflow import convert_to_tensor
from tensorflow import keras
from tensorflow.keras import Layer, Input, backend as K
from tensorflow.keras.losses import Loss
from tensorflow.keras.saving import register_keras_serializable
from RangeDictionaries import MakeHighRange, MakeLowRange


class MineLossFunction0(Loss):
    @register_keras_serializable()
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
    def get_config(self):
        config = super().get_config()
        return config
    def FirstPart(self, y_true, y_pred):
        Alpha = 0.1
        Cond = tf.less(tf.abs(y_true[0]-y_true[1]),0.000001)
        lossTensor = tf.where(Cond,tf.abs(y_pred[0]-y_pred[1]),0.0)

        lossOne = K.mean(K.square((lossTensor)*Alpha))

        return lossOne
    def SecondPart(self, y_pred):
        Beta = 0.025
        HV = MakeHighRange()
        Cond0 = tf.greater(y_pred[0],HV)
        Cond1 = tf.greater(y_pred[1],HV)
        BoolTensor = tf.concat([Cond0,Cond1],axis = 0)
        lossTensor = tf.where(BoolTensor, 1.0, 0.0)
    
        lossTwo = K.mean(K.square((lossTensor)*Beta))

        return lossTwo
    def ThirdPart(self, y_pred):
        Gamma = 0.025
        LV = MakeLowRange()
        Cond0 = tf.less(y_pred[0],LV)
        Cond1= tf.less(y_pred[1],LV)

        BoolTensor = tf.concat([Cond0,Cond1],axis = 0)
        lossTensor = tf.where(BoolTensor, 1.0, 0.0)
        lossThree = K.mean(K.square((lossTensor)*Gamma))
        
        return lossThree
    def MSE(self, y_true, y_pred):
        return K.mean(K.square(y_pred - y_true))
    def MAE(self, y_true, y_pred):
        return  K.mean(K.abs(y_pred - y_true))
    def call(self, y_true, y_pred):

        lossZero = self.MSE(y_true, y_pred)
        lossOne = self.FirstPart(y_true, y_pred)
        lossTwo = self.SecondPart(y_pred)
        lossThree = self.ThirdPart(y_pred)

        return lossZero+lossOne+lossTwo+lossThree
    
    
import tensorflow as tf
import matplotlib.pyplot as plt 
from tensorflow import keras
import numpy as np
import sys
from MLModels2 import QuickBaseModel
import pickle as pick

dir = '../DataFiles/NumpyFiles/500/'
TrainingData = np.load(dir+'rnd_mat_BandStucture.npy')
Results = np.load(dir+'rnd_mat_OpticalParameters.npy')
Shape = TrainingData.shape[1:]


with open("ParamsDict.pkl", 'rb') as f:
    Params = pick.load(f)

SeqModel = QuickBaseModel()

ReturnedModel = SeqModel.GetModel(Shape, "elu")


ReturnedModel.compile(loss = keras.losses.MeanSquaredError(),optimizer = keras.optimizers.Adam(learning_rate=Params["LearningRate"], epsilon=Params["Epsilon"]))
ReturnedModel.summary()
ReturnedModel.fit(TrainingData, Results, epochs = Params["Epoch"], batch_size = Params["BatchSize"])

ReturnedModel.save("model.keras")


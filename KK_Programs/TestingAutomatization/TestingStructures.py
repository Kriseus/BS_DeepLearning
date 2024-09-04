import tensorflow as tf
import time 
import pickle as pick
import numpy as np
import sys
sys.path.append('../DeepLearningSimple')
from MLModels2 import QuickBaseModel, QuickConvModel, QuickDenseModel
from CB import MyCallback 

dir = '../DataFiles/NumpyFiles/500/'
TDa = np.load(dir+'rnd_mat_BandStucture.npy')
Res = np.load(dir+'rnd_mat_OpticalParameters.npy')
# Shape = TrainingData.shape[1:]
TDa = tf.convert_to_tensor(TDa)
Res = tf.convert_to_tensor(Res)



with open("ParamsDict.pkl", 'rb') as f:
    Params = pick.load(f)


def ReWriteDictionary(Dictionary):
    # Optimizer = Dictionary["Optimizer"]
    Epsilons = Dictionary["Epsilon"]
    LRs = Dictionary["LearningRate"]
    B_Size = Dictionary["BatchSize"]
    Epochs = Dictionary["Epochs"]
    Activations = Dictionary["Activation"]

    BigList = []
    HelpDict = {}
    for i in Epsilons:
        for j in Epochs:
            for k in B_Size:
                for ii in LRs:
                    for jj in Activations:
                        HelpDict={"Epsilon":i,"Epoch":j,"BatchSize":k,"LearningRate":ii,"Activation":jj}
                        BigList.append(HelpDict)
                        HelpDict = {}

    return BigList



def TestModel(ListOfArguments, ParamsDictionary, TrainingData, Results, Dir, Filename, Shape = TDa.shape[1:]):
    OneCallback = MyCallback()
    EmptyCallback = MyCallback()
    KeyToLoss = list(ParamsDictionary["Loss"].keys())[0]
    KeyToOptimizer = list(ParamsDictionary["Optimizer"].keys())[0]
    LossFunc = ParamsDictionary["Loss"][KeyToLoss]
    Optimizer = ParamsDictionary["Optimizer"][KeyToOptimizer]
    j=0
    for i in ListOfArguments:
        Activation = i["Activation"]
        LR = i["LearningRate"]
        Epsilon = i["Epsilon"]
        BSize=i["BatchSize"]
        Epoch = i["Epoch"]
        Model = QuickBaseModel().GetModel(Shape, Activation)
        TimeStart = time.time()
        Model.compile(
            loss = LossFunc(),
            optimizer = Optimizer(learning_rate = LR, epsilon = Epsilon)
        )

        Model.fit(x = TrainingData, y = Results, epochs = Epoch, verbose = 1, validation_split = 0.05, batch_size = BSize, callbacks = [OneCallback])
        TimeEnd = time.time()
        Min_Val_loss = min(OneCallback.Val_losses)
        ModelComparison = open(Dir + Filename, "a")
        ModelComparison.write(("Epochs: " + str(i["Epoch"])+" Activation: "+ str(i["Activation"]) + " Batch_Size: " + str(i["BatchSize"]) +
                            " Learning_Rate: " + str(i["LearningRate"]) + " Epsilon: " + str(i["Epsilon"]) + "\nOptimizer: " + KeyToOptimizer +
                            "LossFunc " + KeyToLoss + " Time: "+ str(TimeEnd-TimeStart) + " Min_val_loss: " + str(Min_Val_loss)+"\n" + "#" + "\n"))
        
        with open(Dir+"CallBacks_"+str(j)+"_.npy","wb") as NumFile:
            np.save(NumFile, OneCallback.Val_losses)
        j+=1
        OneCallback=EmptyCallback

ReWritten = ReWriteDictionary(Params)
# ReturnedModel = QuickBaseModel().GetModel(Shape, "elu")
TestModel(ReWritten, Params, TDa ,Res, "", "ComparisonData.dat")

# ReturnedModel.compile(loss = keras.losses.MeanSquaredError(),optimizer = keras.optimizers.Adam(learning_rate=Params["LearningRate"], epsilon=Params["Epsilon"]))
# ReturnedModel.summary()
# ReturnedModel.fit(TrainingData, Results, epochs = Params["Epochs"], batch_size = 64)
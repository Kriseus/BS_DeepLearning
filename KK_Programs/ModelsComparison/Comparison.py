from tensorflow import convert_to_tensor, keras
from tensorflow.keras.models import load_model
from RealMaterials import InAs, BN
import sys
sys.path.append('../Siamese/')
from LossFunctions import MineLossFunction0
from RangeDictionaries import MakeHighRange, MakeLowRange
sys.path.append('../TheDeep')
from MLModels2 import QuickBaseModel

InAs_obj = InAs()
BN_obj = BN()

InAs_BS = InAs_obj.GetBandStructure()
InAs_Pars = InAs_obj.GetParameters()

BN_BS = BN_obj.GetBandStructure()
BN_Pars = BN_obj.GetParameters()



filenameA="../Siamese/model.keras"
filenameB="../TheDeep/model.keras"
ModelC = load_model(filenameA)
ModelB = load_model(filenameB)
ModelA = QuickBaseModel().GetModel(BN_BS.shape[1:],"elu")
ModelA.set_weights(ModelC.get_weights())
ModelA.compile(loss = keras.losses.MeanSquaredError(),optimizer = keras.optimizers.Adam())



def MakePredictions(ModelA, ModelB, InAsStructure, BNStructure):
    PredictedInAS_A = ModelA.predict(InAsStructure)
    PredictedBN_A=ModelA.predict(BNStructure)

    PredictedInAS_B=ModelB.predict(InAsStructure)
    PredictedBN_B=ModelB.predict(BNStructure)

    return [PredictedInAS_A,PredictedInAS_B],[PredictedBN_A,PredictedBN_B]

def CalculateError(Pred,Real):
    Diffs = []
    suma = 0
    Pred=Pred[0]
    Real=Real[0]
    # print(len(Pred),"\n",len(Real))
    for i in range(0,len(Pred)):
        w=(Pred[i]-Real[i])**2

        Diffs.append(w)
        suma+=w
        # print(Pred[i])
    # print(suma)
    return float(suma)

    

def Main(PredInAs, PredBN, RealInAs, RealBN):
    DictionaryOfErrors = {"A_InAs":None,
                          "A_BN":None,
                          "B_InAs":None,
                          "B_BN":None}
    
    DictionaryOfErrors["A_BN"]=CalculateError(PredBN[0], RealBN)
    DictionaryOfErrors["A_InAs"]=CalculateError(PredInAs[0], RealInAs)
    DictionaryOfErrors["B_BN"]=CalculateError(PredBN[1], RealBN)
    DictionaryOfErrors["B_InAs"]=CalculateError(PredInAs[1], RealInAs)

    print(DictionaryOfErrors)
    print(DictionaryOfErrors["A_BN"])
    print(DictionaryOfErrors["A_InAs"])
    print(DictionaryOfErrors["B_BN"])
    print(DictionaryOfErrors["B_InAs"])
    

PredINAS,PredBN = MakePredictions(ModelA,ModelB, InAs_BS, BN_BS)

Main(PredINAS, PredBN, InAs_Pars, BN_Pars)
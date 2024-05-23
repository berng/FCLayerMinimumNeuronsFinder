import numpy as np
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.constraints import NonNeg
import tensorflow.keras.backend as K

from sklearn.datasets import load_digits
import functions
import scipy.stats as ss
from statsmodels.stats.multitest import multipletests

def MakeModel(outputDim,inputDim,hiddenDim1,hiddenDim2,ifsvd=False,iftrain=False, svd_model=1,LayerName='none'):

 classifierInput = keras.Input(shape=(inputDim,), name="classifier_vec_input") 
 vec_features=classifierInput
 vec_features=tf.keras.layers.BatchNormalization(name="batchnorm1")(vec_features)
 act=tf.math.abs
# act=tf.nn.relu


 if not iftrain:
  vec_features = layers.Dense(hiddenDim1, name="transform1", activation=act)(vec_features)
  if ifsvd and LayerName=='transform1':
   vec_features=svd_model(vec_features)

 if not iftrain:
  vec_features = layers.Dense(hiddenDim2, name="transform2", activation=act)(vec_features)
  if ifsvd and LayerName=='transform2':
   vec_features=svd_model(vec_features)

 if not iftrain:
  vec_features = layers.Dense(outputDim, name="transform3", activation=act)(vec_features)
  if ifsvd and LayerName=='transform3':
   vec_features=svd_model(vec_features)


 classifierOutput = layers.Softmax(name="tr2_softmax")(vec_features) 
 classifier_model = keras.Model(
    inputs=[classifierInput],
    outputs=[classifierOutput],
    trainable=True,
    name='classifier',
 )
# classifier_model.summary()
 keras.utils.plot_model(classifier_model, 'model.classifier-'+str(ifsvd)+'+'+str(iftrain)+'.png', show_shapes=True,show_layer_activations=True)
 return classifier_model
#===================================



if __name__=='__main__':
 LayerName='' #sys.argv[1]
 Elastic=-1 #float(sys.argv[2])
 TrainTestSplitSeed=451 #int(sys.argv[3])
 import data_mnist_28x28 as data
 CrossValSplit=3

 Xtest,Ytest=data.Xtest,data.Ytest
 Xtr,Ytr=data.Xtr,data.Ytr
 Xval,Yval=data.Xval,data.Yval

 Xfl,Yfl=np.concatenate([Xtr,Xval]),np.concatenate([Ytr,Yval]) ### Full search - overtrained

 evals=[]
 for cv1 in range(CrossValSplit):
  for cv2 in range(CrossValSplit):
#  len=Xfl.shape[0]
#  X1v=Xfl[len*cv//CrossValSplit:len*(cv+1)//CrossValSplit]
#  Y1v=Yfl[len*cv//CrossValSplit:len*(cv+1)//CrossValSplit]
#  X1tr=np.concatenate([Xfl[:len*cv//CrossValSplit],Xfl[len*(cv+1)//CrossValSplit:]])
#  Y1tr=np.concatenate([Yfl[:len*cv//CrossValSplit],Yfl[len*(cv+1)//CrossValSplit:]])
   model1=tf.keras.models.load_model('initial_model_'+str(cv1))
   Y1=model1.predict(Xtest,verbose=False) 
   model2=tf.keras.models.load_model('minimal_model_'+str(cv2))
   Y2=model2.predict(Xtest,verbose=False) 
   q=functions.getQ(Y1,Y2)
   print(cv1,cv2,q)
 quit()

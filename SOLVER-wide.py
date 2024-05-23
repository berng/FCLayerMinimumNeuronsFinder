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
 print('usage: %s layername(transform1/transform2) layerwidth(300)'%(sys.argv[0]))
 LayerName=sys.argv[1]
 Elastic=float(sys.argv[2])
 TrainTestSplitSeed=451 #int(sys.argv[3])
 import data_mnist_28x28 as data
 CrossValSplit=3

 Xtest,Ytest=data.Xtest,data.Ytest
 Xtr,Ytr=data.Xtr,data.Ytr
 Xval,Yval=data.Xval,data.Yval

 Xfl,Yfl=np.concatenate([Xtr,Xval]),np.concatenate([Ytr,Yval]) ### Full search - overtrained
# Xsvd,Ysvd=Xtr,Ytr  ### Use to find full SVD
 Xsvd,Ysvd=Xfl,Yfl  ### Use to estimate correctness
# Xsrch,Ysrch=Xtr,Ytr  ### Use to estimate correctness
 Xsrch,Ysrch=Xfl,Yfl  ### Search at val
# print('Xfl sh',Xfl.shape[0],Xtr.shape)

 inDim=Xtr.shape[1]
 hidDim=int(Elastic) #int(Xtr.shape[1]*Elastic)
 print('hidDim',hidDim)
 outDim=10
# print('================= STATE 1: train models ===================')
 evals=[]
 for cv in range(CrossValSplit):
  len=Xfl.shape[0]
  X1v=Xfl[len*cv//CrossValSplit:len*(cv+1)//CrossValSplit]
  Y1v=Yfl[len*cv//CrossValSplit:len*(cv+1)//CrossValSplit]
  X1tr=np.concatenate([Xfl[:len*cv//CrossValSplit],Xfl[len*(cv+1)//CrossValSplit:]])
  Y1tr=np.concatenate([Yfl[:len*cv//CrossValSplit],Yfl[len*(cv+1)//CrossValSplit:]])
  model=MakeModel(outDim,inDim,hidDim,hidDim)
  LR=1e-3
#  for LR in (1e-3,1e-4,1e-5):
  if True:
   model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),loss='categorical_crossentropy',metrics='accuracy')
   es=tf.keras.callbacks.EarlyStopping(monitor='val_loss',mode='min',patience=3,restore_best_weights=True)
   history=model.fit(X1tr,Y1tr,validation_data=[X1v,Y1v],epochs=1000,callbacks=[es],verbose=False)
  print('eval test')
  eval_hist=model.evaluate(Xtest,Ytest,verbose=True)  # change to X1v,Y1v
  print('eval val')
  eval_hist=model.evaluate(X1v,Y1v,verbose=True)  # change to X1v,Y1v
  evals.append(eval_hist)
  model.save('initial_model_'+str(cv))
 if LayerName=='none':
  quit()
 del model
 evals=np.array(evals)
# print('================= STATE 2: a. Get Dim ===================')

# hidDim=160 #Xtr.shape[1]*5
# hidDim=129 #Xtr.shape[1]*5
 SEARCH_STAGE=0
 START_DIM=40
# if SEARCH_STAGE==0:

 len=Xfl.shape[0]

 old_model=tf.keras.models.load_model('initial_model_'+str(0),compile=False)
 old_model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=['accuracy']
              )
 inp=old_model.input
 outp=old_model.get_layer(LayerName).output
 temp_model=tf.keras.models.Model(inputs=inp,outputs=outp)
 LayerOutputs=temp_model.predict(Xsvd[:1],verbose=False)
 LayerDim=LayerOutputs.shape[1]


# print('================= STATE 2: Search opt number fo neurons ===================')

 finalResults=[]
 i1=0
 i2=0
 for i1 in range(CrossValSplit):
  for i2 in range(CrossValSplit):
   if i1==i2: continue

   testedLayerMin=1
   testedLayerMax=LayerDim
   MaxPv=1
   MinPv=0
   alpha=0.95

   for itt in range(20):
    testedLayerDim=(testedLayerMax+testedLayerMin)//2

    idxs=np.array(range(len))
    idxs_val={}
    idxs_train={}
    models1={}
    models2={}
    for cv in range(CrossValSplit):

# get train and val for this CV
     idxs_val[cv]=idxs[len*cv//CrossValSplit:len*(cv+1)//CrossValSplit]
     idxs_train[cv]=np.concatenate([idxs[:len*cv//CrossValSplit],idxs[len*(cv+1)//CrossValSplit:]])

     X1v=Xfl[idxs_val[cv]]
     Y1v=Yfl[idxs_val[cv]]
     X1tr=Xfl[idxs_train[cv]]
     Y1tr=Yfl[idxs_train[cv]]

# ===load old model 
     old_model=tf.keras.models.load_model('initial_model_'+str(cv),compile=False)
     old_model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=['accuracy']
              )
     models1[cv]=old_model
# == make new models ===

     inp=old_model.input
     outp=old_model.get_layer(LayerName).output
     outp_hidden=old_model.get_layer('transform1').output
     temp_model=tf.keras.models.Model(inputs=inp,outputs=outp)
     LayerOutputs=temp_model.predict(Xsvd,verbose=False)
     LayerDim=LayerOutputs.shape[1] #-10
     for testedLayerDim2 in ((LayerDim,testedLayerDim)):
      Transform,InvTransform=functions.findSvdTransform(LayerOutputs,testedLayerDim2)
     svd_model=functions.SVD_Model(LayerDim,testedLayerDim)
# svd_model.summary()
     svd_model=functions.ReinitedSVD(LayerDim,testedLayerDim,Transform,InvTransform,svd_model)
     svd_model.trainable=False
     pred=svd_model.predict(LayerOutputs[:1,:],verbose=False)
     model=MakeModel(Ytr.shape[1],inp.shape[1],outp_hidden.shape[1],outp_hidden.shape[1],ifsvd=True,svd_model=svd_model,LayerName=LayerName,iftrain=False)
     model=functions.copyWeights(old_model,model)
     model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=['accuracy']
              )
     models2[cv]=model


 # calculate stat equivalence: Q distributions for M1,M1 and M1,M2
    Q123=functions.getStatEquivalence(models1,models2,Xfl,idxs_train,Yfl,i1,i2,alpha=alpha)

    Qq3=Q123

#    print(testedLayerDim,'Qstats:',Qq3,end='\n')

#  print(flush=True)
    Q2reach=(1.+np.array(evals)[:,1].mean())/2.
#    Q2reach=np.array(evals)[:,1].mean()
#    print('Q2reach:',Q2reach)
#    quit()
    if Qq3>=Q2reach and testedLayerDim<testedLayerMax:
     testedLayerMax=testedLayerDim
    if Qq3<Q2reach and testedLayerDim>testedLayerMin:
     testedLayerMin=testedLayerDim
    if testedLayerMax<testedLayerMin+2:
     break
    old=testedLayerDim
   
   print(i1,i2,' ', hidDim,' result:',testedLayerMax,'Q2reach',Q2reach,flush=True)
   finalResults.append(testedLayerMax)
finalResults=np.array(finalResults)
print('TTSS:',TrainTestSplitSeed,hidDim,' result:',finalResults.mean(),finalResults.std(),evals[:,1])

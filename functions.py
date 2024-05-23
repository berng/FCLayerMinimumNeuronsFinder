import numpy as np
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.constraints import NonNeg
import tensorflow.keras.backend as K
# import tensorflow_addons as tfa
from sklearn.decomposition import PCA


#decClassNum=int(sys.argv[1])


def SVD_Model(totalClassNum,decClassNum):
 pcaInput = keras.Input(shape=(totalClassNum,), name="pca_input") 

 x=layers.Reshape((totalClassNum,1))(pcaInput)
 cl,zeroInput = tf.split(pcaInput, [totalClassNum-1,1], 1)
 zeroInput=tf.cast(zeroInput*0,dtype=tf.dtypes.int32)

 pcaMatrixEmbLayer=layers.Embedding(2,totalClassNum*decClassNum,name='svd')
 pcaMatrixInvEmbLayer=layers.Embedding(2,totalClassNum*decClassNum,name='svdInv')

 pcaMatrixEmb=pcaMatrixEmbLayer(zeroInput)
 pcaMatrixInvEmb=pcaMatrixInvEmbLayer(zeroInput)
 pcaMatrixEmb=layers.Reshape((totalClassNum,decClassNum))(pcaMatrixEmb)
 pcaMatrixInvEmb=layers.Reshape((decClassNum,totalClassNum))(pcaMatrixInvEmb)

 x = layers.Dot(axes=(1,1))([pcaMatrixEmb,x])
 x = layers.Dot(axes=(1,1))([pcaMatrixInvEmb,x])
 outputs =layers.Flatten()(x)
 svd_model=tf.keras.models.Model(inputs=[pcaInput],outputs=[outputs],name='SVD_autoencoder')
 return svd_model


def findSvdTransform(X,hidShape,recall=True):
 inShape=X.shape[1]
 if hidShape==inShape or not recall:
  U, S, Vh = np.linalg.svd(X, full_matrices=False)
  if recall:
   np.savez('svd_res.npz',u=U,s=S,vh=Vh)
 else:
  dd=np.load('svd_res.npz')
  U=dd['u']
  S=dd['s']
  Vh=dd['vh']

 InvTransform=np.dot(np.diag(S),Vh)[:hidShape,:]
 Transform=np.linalg.pinv(InvTransform[:hidShape,:])
 return Transform,InvTransform

def ReinitedSVD(inShape,hidShape,Transform,InvTransform,svd_model):
 weights=np.array([Transform.reshape(inShape*hidShape),np.zeros(inShape*hidShape)])
# weights=np.array([Transform.reshape(inShape*hidShape),Transform.reshape(inShape*hidShape)])
 svd_model.get_layer('svd').set_weights([weights])
 svd_model.get_layer('svdInv').set_weights([np.array([InvTransform.reshape(inShape*hidShape),np.zeros(inShape*hidShape)])])
# svd_model.get_layer('svdInv').set_weights([np.array([InvTransform.reshape(inShape*hidShape),InvTransform.reshape(inShape*hidShape)])])
 return svd_model 


def copyWeights(src_model,dest_model):
 for name in ('batchnorm1','transform1','transform2','transform3'):
  dest_model.get_layer(name).set_weights(src_model.get_layer(name).weights)
 return dest_model

 
def intersection(np1, np2):
    lst1=list(np1)
    lst2=list(np2)
    lst3 = [value for value in lst1 if value in lst2]
    return np.array(lst3) 

def getTestSet(np1, np2):
    lst1=list(np1)
    lst2=list(np2)
    lst3 = [value for value in lst1 if value not in lst2]+[value for value in lst2 if value not in lst1]
#    lst3 = [value for value in lst2]
#    lst3 = [value for value in lst1]
#    lst3 = [value for value in lst1 if value in lst2]
    return np.array(lst3) 

from sklearn.metrics import accuracy_score
def getQ(yt,yp,alpha=0.05):
 res=[]
 yt1=np.argmax(yt,axis=-1)
 yp1=np.argmax(yp,axis=-1)
 for i in range(10000):
  idx=np.random.randint(0,yp1.shape[0]-1,yp1.shape[0])
  res.append(accuracy_score(yt1[idx],yp1[idx]))
 res=np.array(res)
# per=np.percentile(res,100.0*(1.-alpha))
 per=res.min()
 return per

def getStatEquivalence(models1,models2,X,crossval_idxs,Y,i1,i2,alpha=0.05):
  Qs=[]
# for i1,j1 in enumerate(models1.keys()):
# for i2,j2 in enumerate(models2.keys()):
#   if i1!=i2:
#    model1=models1[i1]
  model1b=models1[i2]
  model2=models2[i2]
  test_set=getTestSet(crossval_idxs[i1],crossval_idxs[i2])
#    Y1=model1.predict(X[test_set],verbose=False)
  Y2=model1b.predict(X[test_set],verbose=False)
  Y3=model2.predict(X[test_set],verbose=False)
  Q23max=getQ(Y2,Y3,alpha=alpha)
#  Qs.append(Q23max)
#  Qs=np.array(Qs)
# print('Qs:',Qs[:,0].min(),Qs[:,1].min(),Qs[:,2].min())
  return Q23max



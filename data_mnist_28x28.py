import numpy as np
import sys
import tensorflow as tf
from sklearn.model_selection import train_test_split

mnist = tf.keras.datasets.mnist

(Xtrain, Ytrain), (Xtest, Ytest) = mnist.load_data()

Xtrain=tf.expand_dims(Xtrain,axis=-1)
Xtest=tf.expand_dims(Xtest,axis=-1)

#Xtrain=tf.nn.max_pool(Xtrain,ksize=4,strides=4,padding='SAME')
#Xtrain=tf.pad(Xtrain,[[0,0],[1,0],[1,0],[0,0]])
#Xtest=tf.nn.max_pool(Xtest,ksize=4,strides=4,padding='SAME')
#Xtest=tf.pad(Xtest,[[0,0],[1,0],[1,0],[0,0]])


Xtrain=tf.reshape(Xtrain,(Xtrain.shape[0],Xtrain.shape[1]*Xtrain.shape[2]))
Xtest=tf.reshape(Xtest,(Xtest.shape[0],Xtest.shape[1]*Xtest.shape[2]))
Xtrain=np.array(Xtrain)
Xtest=np.array(Xtest)

Ytrain=tf.keras.utils.to_categorical(Ytrain)
Ytest=tf.keras.utils.to_categorical(Ytest)

TrainTestSplitSeed=451 #int(sys.argv[3])
print('TTS:',TrainTestSplitSeed)
Xtr,Xval,Ytr,Yval=train_test_split(Xtrain,Ytrain,random_state=TrainTestSplitSeed)
#Xtr,Xval,Ytr,Yval=Xtr[:10000],Xval[:10000],Ytr[:10000],Yval[:10000]

if __name__=='__main__':
 print(Xtrain.shape)
 print(Ytrain.shape)

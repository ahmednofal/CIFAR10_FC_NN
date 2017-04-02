from keras.models import load_model
import numpy as np
from keras import optimizers

import utils
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import LeakyReLU, ELU, PReLU,ThresholdedReLU
from keras.activations import relu, tanh, elu
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras.constraints import maxnorm

from keras.callbacks import ModelCheckpoint, TensorBoard
nb_classes = 10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

#X_train = X_train.reshape(X_train.shape[0], -1).astype('float32')/255
#X_test = X_test.reshape(X_test.shape[0], -1).astype('float32')/255

X_train = X_train / 255
X_test = X_test / 255

mean = np.mean(X_train)

X_train -= mean
X_test -= mean

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

#model = Sequential()

model = load_model("/home/m.n.elnokrashy/Downloads/Saved_Data/weights.h5")


y_test_pred = model.predict_classes(X_test)


classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
ccrn = []

for i in range(10):
        numCorrect = np.sum(y_test_pred[x] == y_test[x] and y_test_pred[x] == i for x in range(X_test.shape[0]))
        ccrn.append(float(numCorrect) / 1000)
        print ('CCRn of %s is %d / %d with correct of %f' % (classes[i], ccrn[i]*1000,1000, ccrn[i]))
print("\n",ccrn,"\n")
accuracy = 0
for z in range(10):
	accuracy += ccrn[z]

print ('Got %d / %d correct => ACCR accuracy: %f' % (accuracy*1000, 10000, accuracy/10))


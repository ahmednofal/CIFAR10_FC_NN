import numpy as np
from keras import optimizers

np.random.seed(1337)


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
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, TensorBoard

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

nb_classes = 10
shift = 0.1

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

datagen = ImageDataGenerator(featurewise_center= False,
 samplewise_center=False,
 featurewise_std_normalization=False,
 samplewise_std_normalization=False,
 zca_whitening=False,
 rotation_range=0,
 width_shift_range=shift,
 height_shift_range=shift,
 horizontal_flip=True,
 vertical_flip=False,
 )
X_train = X_train / 255
X_test = X_test / 255

mean = np.mean(X_train)

X_train -= mean
X_test -= mean

datagen.fit(X_train)



model = Sequential()

inputShape = X_train.shape[1:]
model.add(Flatten(input_shape = inputShape))

#kernel_constraint=maxnorm(4),

model.add(Dense(1500,
input_shape = X_train.shape[1:],
bias_initializer='zeros'
))

model.add(BatchNormalization())
model.add(PReLU())
model.add(Dropout(0.5))

model.add(Dense(750))
model.add(BatchNormalization())
model.add(PReLU())
model.add(Dropout(0.4))

model.add(Dense(150))
model.add(BatchNormalization())
model.add(PReLU())
model.add(Dropout(0.3))

model.add(Dense(10))
model.add(BatchNormalization())
model.add(Activation('softmax'))
sgd = optimizers.SGD(lr=0.2, momentum=0.95, decay=4e-3, nesterov=True)
Nadam = optimizers.Nadam(lr = 0.000002,schedule_decay = 0.000002/1000)
adam = optimizers.Adam(lr = 0.0001,decay = 0.0001/1000)

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

batchS = 128

checkpointer = ModelCheckpoint(filepath="/home/m.n.elnokrashy/Downloads/Saved_Data/weights.h5", verbose=0   , save_best_only=True)
graph_save = TensorBoard(log_dir="/home/m.n.elnokrashy/Downloads/Saved_Data/", histogram_freq=0, write_graph=True, write_images=False)


model.fit_generator(datagen.flow(X_train, Y_train, batch_size = 128),
                    steps_per_epoch=X_train.shape[0]/128,
                    epochs=800,
                    validation_data=(X_test, Y_test),
		    callbacks=[graph_save,checkpointer])


y_test_pred = model.predict_classes(X_test)


classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
ccrn_array = []

for z in range(10):
	ccrn_array[z] = float((np.sum(y_test_pred == z)/1000))
	print ('%s CCRn accuracy is %d / %d correct => : %f' % (classes[z], ccrn_array[z]*1000,1000, ccrn_array[z]))
num_correct = np.sum(y_test_pred == y_test)
accuracy = float((num_correct) / 10000)
print ('Got %d / %d correct => ACCR accuracy: %f' % (num_correct,10000, accuracy))




import tensorflow as tf


import numpy as np 
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras import losses
from tensorflow.keras import optimizers
from keras.layers.core import Dense, Dropout, Activation, Flatten
from features2arr import txt2Arr

# 1.
path0 = '0-training.txt'
path1 = '0-testing.txt'

labelTrain = txt2Arr(path0,"label")
vectorTrain = txt2Arr(path0,"vector")

labelTest = txt2Arr(path1,"label")
vectorTest = txt2Arr(path1,"vector")


nb_classes = 1 #we have these many digits in our training

# 1. Prepare data 
trainX = np.array(vectorTrain)
trainY = np.array(labelTrain)

testX = np.array(labelTest)
testY = np.array(vectorTest)

print('Prepare data completed')

# 2. Build model 
model = Sequential()
model.add(Dense(1, input_shape=((None, 2560))))
# model.add(Dense(128))
# model.add(Dense(1, input_shape=((None, 2560))))
# model.add(Activation('linear'))
# model.add(Activation(activation = "softmax"))
# model.add(Dense(64))
# model.add(Dense(nb_classes,name="outputlayer"))
# model.add(Dense(32)) 
# model.add(Activation('softmax'))
# model.add(Dense(16,activation='softmax'))
# model.add(Dense(8))
# model.add(Flatten())
# model.add(Dense(4))
# model.add(Dense(nb_classes, activation='softmax'))
# model.add(Dense(units=128, input_shape=(None,2560), activation='softmax'))
# model.add(Dense(units=128, activation='softmax'))
# model.add(Dropout(0.25))
# model.add(Dense(units=6, activation='softmax'))

print('Created an model!')

# 3. gradient descent optimizer and loss function 
sgd = optimizers.SGD(lr=0.05)
#model.compile(loss=losses.binary_crossentropy, optimizer=sgd)
model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
# 4. Train the model 
model.fit(trainX, trainY,validation_data=(testX,testY) ,epochs=20, batch_size=1, verbose=2) 
print("Train complete");
#
#Test the model
#
print("Testing on test data")
(loss, accuracy) = model.evaluate(
    testX, 
    testY,
    batch_size = 1, 
    verbose = 1)

# Print the model's accuracy
print("Accuracy="+ str(accuracy))
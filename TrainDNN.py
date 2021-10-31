import tensorflow as tf
import numpy as np 
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras import losses
from keras import utils
from tensorflow.keras import optimizers
from keras.layers.core import Dense, Dropout, Activation, Flatten
from features2arr import txt2Arr
from tensorflow.keras.utils import to_categorical

# 1.
path0 = '1-testing-dave.txt'
path1 = '1-training-dave.txt'

nb_classes = 10 #we have these many digits in our training

labelTrain = txt2Arr(path0,"label")
vectorTrain = txt2Arr(path0,"vector")
labelTest = txt2Arr(path1,"label")
vectorTest = txt2Arr(path1,"vector")
trainX = np.array(vectorTrain)
trainY = np.array(labelTrain)
testX = np.array(vectorTest)
testY = np.array(labelTest)
trainY_one_hot=to_categorical(trainY) # convert Y into an one-hot vector
testY_one_hot=to_categorical(testY) # convert Y into an one-hot vector

print('Prepare data completed')

# 2. Build model 
model = Sequential()
model.add(Flatten(name="layer1"))
model.add(Dense(1024))
model.add(Dense(2048))
model.add(Dense(4096))
model.add(Dense(2048))
model.add(Dense(2048))
model.add(Dense(1024))
model.add(Dense(512))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(64))
model.add(Activation(activation = "softmax"))
model.add(Dense(nb_classes)) 
model.add(Activation("softmax",name="outputlayer"))


model.compile(
    loss = "categorical_crossentropy", 
    optimizer = 'adam',
    metrics = ["accuracy"])
print("Compilation complete");


model.fit(
    trainX, 
    trainY_one_hot, 
    batch_size = 128, 
    epochs = 20,
	verbose = 1)
print("Train complete");


# Test the model
print("Testing on test data")
(loss, accuracy) = model.evaluate(
    testX, 
    testY_one_hot,
    batch_size = 10, 
    verbose = 1)

# Print the model's accuracy
print("Lost= "+ str(loss) + " Accuracy= "+ str(accuracy))
model.save('saveDNN.h5')
model.summary()





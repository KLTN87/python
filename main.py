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
path0 = 'training-dave.txt'
path1 = 'testing-dave.txt'

labelTrain = txt2Arr(path0,"label")
vectorTrain = txt2Arr(path0,"vector")

labelTest = txt2Arr(path1,"label")
vectorTest = txt2Arr(path1,"vector")


nb_classes = 62 #we have these many digits in our training
input_shape = (None, 2560)

# 1. Prepare data 
trainX = np.array(vectorTrain)
trainY = np.array(labelTrain)


testX = np.array(vectorTest)
testY = np.array(labelTest)




# training_set_shape = trainX.shape
# testing_set_shape = trainY.shape
# print(training_set_shape)
# print(testing_set_shape)


print('Prepare data completed')

# 2. Build model 
model = Sequential()
model.add(Flatten())
model.add(Dense(128))
model.add(Activation(activation = "softmax"))
model.add(Dense(nb_classes,name="outputlayer")) 
model.add(Activation("softmax"))

trainY_one_hot=to_categorical(trainY) # convert Y into an one-hot vector

testY_one_hot=to_categorical(testY) # convert Y into an one-hot vector

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

sgd = optimizers.SGD(lr=0.01)
model.compile(
    loss = "categorical_crossentropy", 
    optimizer = 'adam',
    metrics = ["accuracy"])
print("Compilation complete");




model.fit(
    trainX, 
    trainY_one_hot, 
    batch_size = 10, 
    epochs = 20,
	verbose = 1)
print("Train complete");






# # Test the model

print("Testing on test data")
(loss, accuracy) = model.evaluate(
    testX, 
    testY_one_hot,
    batch_size = 10, 
    verbose = 2)

# Print the model's accuracy
print("Accuracy="+ str(accuracy))

# print(trainX)

# print("One hot")
# print(trainY_one_hot)

# print("Normal")

# print(trainY)


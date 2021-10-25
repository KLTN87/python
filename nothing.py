import tensorflow as tf


import numpy as np 
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras import losses
from tensorflow.keras import optimizers
from keras.layers.core import Dense, Dropout, Activation, Flatten
from features2arr import txt2Arr

# 1.
path0 = '0.txt'
path1 = 'testing.txt'

labelTrain = txt2Arr(path0,"label")
vectorTrain = txt2Arr(path0,"vector")


nb_classes = 1 #we have these many digits in our training

# 1. Prepare data 
X = np.array(vectorTrain)
y = np.array(labelTrain)



# 2. Build model 
model = Sequential()
# model.add(Dense(1, input_shape=(1,2304)))
# model.add(Activation('sigmoid'))


# Flatten the network
model.add(Flatten())
# Add a fully-connected hidden layer
model.add(Dense(128, input_shape=(1,2304)))
model.add(Activation('sigmoid'))
model.add(Activation(activation = "softmax"))
# Add a fully-connected output layer - the output layer nodes should match the count of image classes
model.add(Dense(nb_classes,name="outputlayer")) 
# Add a softmax activation function
model.add(Activation("softmax"))
#
#Display Summary
#
model.summary()


# Compile the network
model.compile(
    loss = "categorical_crossentropy", 
    optimizer = optimizers.SGD(lr = 0.01),
    metrics = ["accuracy"])
print("Compilation complete");
print("Train begin");
# Train the model 

total_epochs=20

model.fit(
    X, 
    y, 
    batch_size = 128, 
    epochs = total_epochs,
	  verbose = 1)
print("Train complete");
#
#Test the model
#
# print("Testing on test data")
# (loss, accuracy) = model.evaluate(
#     X, 
#     y,
#     batch_size = 128, 
#     verbose = 1)

# # Print the model's accuracy
# print("Accuracy="+ str(accuracy))

# print("Train complete");
# #
# #Test the model
# #
# print("Testing on test data")
# (loss, accuracy) = model.evaluate(
#     test_data1, 
#     test_target1,
#     batch_size = 128, 
#     verbose = 1)

# # Print the model's accuracy
# print("Accuracy="+ str(accuracy))

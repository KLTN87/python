import numpy as np
np.random.seed(2016)

from tensorflow import keras
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import numpy as np



from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import model_from_json
from ImageLoader import load_images,ReShapeData
from keras import backend as K

#
#The following variables should be set to the folder where MNIST images have been extracted 
#
mnist_train_path_full="C:\\Users\\Admin\\Downloads\\mnist_png\\training\\*\\*.png"
mnist_test_path="C:\\Users\\Admin\\Downloads\\mnist_png\\testing\\*\\*.png"
nb_classes = 10 #we have these many digits in our training

#
#Load training images
#
print("Loading training images")
(train_data, train_target)=load_images(mnist_train_path_full)
(train_data1,train_target1)=ReShapeData(train_data,train_target,nb_classes)
print('Shape:', train_data1.shape)
print(train_data1.shape[0], ' train images were loaded')
#
#Load test images
#
print("Loading testing images")
(test_data, test_target)=load_images(mnist_test_path)
(test_data1,test_target1)=ReShapeData(test_data,test_target,nb_classes)
print('Shape:', test_data1.shape)
print(test_data1.shape[0], ' test images were loaded')
print("Load complete")
# 
# Create a sequential model
#
model = Sequential()

# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(nb_classes,name="outputlayer")) 
# model.add(Activation("softmax"))

# Lost= 0.06838732957839966 Accuracy= 0.9810000061988831


model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(nb_classes, activation='softmax'))

# Lost= 0.05874831601977348 Accuracy= 0.9868000149726868


# #convolutional layer with rectified linear unit activation
# model.add(Convolution2D(32, kernel_size=(3, 3),
#                  activation='relu',
#                  input_shape=(28, 28, 1)))
# #32 convolution filters used each of size 3x3
# #again
# model.add(Convolution2D(64, (3, 3), activation='relu'))
# #64 convolution filters used each of size 3x3
# #choose the best features via pooling
# model.add(MaxPooling2D((2, 2)))
# #randomly turn neurons on and off to improve convergence
# model.add(Dropout(0.25))
# #flatten since too many dimensions, we only want a classification output
# model.add(Flatten())
# #fully connected to get all relevant data
# model.add(Dense(128, activation='relu'))
# #one more dropout for convergence' sake :) 
# model.add(Dropout(0.5))
# #output a softmax to squash the matrix into output probabilities
# model.add(Dense(nb_classes, activation='softmax'))

# # Lost= 0.0335056334733963 Accuracy= 0.9926000237464905




# Compile the network
model.compile(
    loss = "categorical_crossentropy", 
    optimizer = 'adam',
    metrics = ["accuracy"])
print("Compilation complete");
print("Train begin");
# Train the model 

total_epochs=20

model.fit(
    train_data1, 
    train_target1, 
    batch_size = 128, 
    epochs = total_epochs,
	  verbose = 1)
print("Train complete");
#
#Test the model
#
print("Testing on test data")
(loss, accuracy) = model.evaluate(
    test_data1, 
    test_target1,
    batch_size = 128, 
    verbose = 1)

# Print the model's accuracy
print("Lost= "+ str(loss) + " Accuracy= "+ str(accuracy))
model.save('saveCNN.h5')
model.summary()



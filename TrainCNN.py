import numpy as np
np.random.seed(2016)
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from ImageLoader import load_images,ReShapeData
from keras import backend as K

#
#The following variables should be set to the folder where MNIST images have been extracted 
#
train_path_full="C:\\Users\\Admin\\Downloads\\mnist_png\\training\\*\\*.png"
test_path="C:\\Users\\Admin\\Downloads\\mnist_png\\testing\\*\\*.png"
nb_classes = 10 #we have these many digits in our training

#
#Load training images
#
print("Loading training images")
(train_data, train_target)=load_images(train_path_full)
(train_data1,train_target1)=ReShapeData(train_data,train_target,nb_classes)
print('Shape:', train_data1.shape)
print(train_data1.shape[0], ' train images were loaded')
#
#Load test images
#
print("Loading testing images")
(test_data, test_target)=load_images(test_path)
(test_data1,test_target1)=ReShapeData(test_data,test_target,nb_classes)
print('Shape:', test_data1.shape)
print(test_data1.shape[0], ' test images were loaded')
print("Load complete")
# 
# Create a sequential model
#
model = Sequential()
# Add the first convolution layer
model.add(Convolution2D(
    name="conv1",
    filters = 20,
    kernel_size = (5, 5),
    padding = "same",
    input_shape = (28, 28, 1)))
# Add a ReLU activation function
model.add(Activation(
    activation = "relu"))
# Add a pooling layer
model.add(MaxPooling2D(
    name="maxpool1",
    pool_size = (2, 2),
    strides =  (2, 2)))
# Add the second convolution layer
model.add(Convolution2D(
    name="conv2",
    filters = 50,
    kernel_size = (5, 5),
    padding = "same"))
# Add a ReLU activation function
model.add(Activation(
    activation = "relu"))
# Add a second pooling layer
model.add(MaxPooling2D(
    name="maxpool2",
    pool_size = (2, 2),
    strides = (2, 2)))
# Flatten the network
model.add(Flatten())
# Add a fully-connected hidden layer
model.add(Dense(500))
# Add a ReLU activation function
model.add(Activation(activation = "relu"))
# Add a fully-connected output layer - the output layer nodes should match the count of image classes
model.add(Dense(nb_classes,name="outputlayer")) 
# Add a softmax activation function
model.add(Activation("softmax"))



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


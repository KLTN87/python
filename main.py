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

labelTrain = txt2Arr(path0,"label")
vectorTrain = txt2Arr(path0,"vector")

labelTest = txt2Arr(path1,"label")
vectorTest = txt2Arr(path1,"vector")


nb_classes = 10 #we have these many digits in our training
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
model.add(Dense(256))
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
    batch_size = 128, 
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


# print("One hot")
# print(trainY_one_hot)
# print("Normal")
# print(trainY)


# Test predict 
from random import randrange
lenTextX = len(testX)
rd = randrange(lenTextX)
predict_x = model.predict(testX)[rd] 
classes_x = np.argmax(predict_x, axis=-1)
print("Test predict")
print(classes_x)
print(testY[rd])



# convert to .pb

from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

# Convert Keras model to ConcreteFunction
full_model = tf.function(lambda x: model(x))
full_model = full_model.get_concrete_function(
    x=tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

# Get frozen ConcreteFunction
frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()

# inspect the layers operations inside your frozen graph definition and see the name of its input and output tensors
layers = [op.name for op in frozen_func.graph.get_operations()]
print("-" * 50)
print("Frozen model layers: ")
for layer in layers:
    print(layer)

print("-" * 50)
print("Frozen model inputs: ")
print(frozen_func.inputs)
print("Frozen model outputs: ")
print(frozen_func.outputs)

# Save frozen graph from frozen ConcreteFunction to hard drive
# serialize the frozen graph and its text representation to disk.
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir="./frozen_models",
                  name="simple_frozen_graph.pb",
                  as_text=False)

#Optional
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir="./frozen_models",
                  name="simple_frozen_graph.pbtxt",
                as_text=True)

model.summary()
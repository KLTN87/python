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

print('Prepare data completed')

# 2. Build model 
model = Sequential()
model.add(Flatten(name="layer1"))
model.add(Dense(256))
model.add(Activation(activation = "softmax"))
model.add(Dense(nb_classes)) 
model.add(Activation("softmax",name="outputlayer"))

trainY_one_hot=to_categorical(trainY) # convert Y into an one-hot vector

testY_one_hot=to_categorical(testY) # convert Y into an one-hot vector


# sgd = optimizers.SGD(lr=0.01)
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


# print("One hot")
# print(trainY_one_hot)
# print("Normal")
# print(trainY)


# Test predict 
# from random import randrange
# lenTextX = len(testX)
# rd = randrange(lenTextX)
# predict_x = model.predict(testX)[rd] 
# classes_x = np.argmax(predict_x, axis=-1)
# print("Test predict")
# print(classes_x)
# print(testY[rd])




model.save('save.h5')

# model.save('save')

model.summary()










# full_model = tf.function(lambda inputs: model(inputs))    
# full_model = full_model.get_concrete_function([tf.TensorSpec(model_input.shape, model_input.dtype) for model_input in model.inputs])


# input_names = [inp.name for inp in full_model.inputs]
# output_names = [out.name for out in full_model.outputs]
# print("Inputs:", input_names)
# print("Outputs:", output_names)

# from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
# frozen_func = convert_variables_to_constants_v2(full_model)
# frozen_func.graph.as_graph_def()

# from tf2onnx import tf_loader
# from tf2onnx.tfonnx import process_tf_graph
# from tf2onnx.optimizer import optimize_graph
# from tf2onnx import utils, constants
# from tf2onnx.handler import tf_op
# extra_opset = [utils.make_opsetid(constants.CONTRIB_OPS_DOMAIN, 1)]
# with tf.Graph().as_default() as tf_graph:
#     tf.import_graph_def(frozen_func.graph.as_graph_def(), name='')
# with tf_loader.tf_session(graph=tf_graph):
#     g = process_tf_graph(tf_graph, input_names=input_names, output_names=output_names, extra_opset=extra_opset)
# onnx_graph = optimize_graph(g)
# model_proto = onnx_graph.make_model("converted")
# utils.save_protobuf("model2b.onnx", model_proto)
# print("Conversion complete!")
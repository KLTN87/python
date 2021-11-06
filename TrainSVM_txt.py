import numpy as np
from sklearn import svm, metrics
from features2arr import txt2Arr
import pickle

path0 = '1-testing-dave.txt'
path1 = '1-training-dave.txt'

print("TRAIN")

labelTrain = txt2Arr(path0,"label")
vectorTrain = txt2Arr(path0,"vector")
labelTest = txt2Arr(path1,"label")
vectorTest = txt2Arr(path1,"vector")
trainX = np.array(vectorTrain)
trainY = np.array(labelTrain)
testX = np.array(vectorTest)
testY = np.array(labelTest)


model = svm.SVC()
model.fit(trainX, trainY)


print("PREDICT")
predict = model.predict(testX)

print("RESULT")
ac_score = metrics.accuracy_score(testY, predict)
cl_report = metrics.classification_report(testY, predict)
print("Score = ", ac_score)
print(cl_report)


filename = 'TrainSVM_txt.sav'
pickle.dump(model, open(filename, 'wb'))
 
# some time later...
 
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(testX, testY)
print(result)
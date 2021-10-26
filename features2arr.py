import numpy

def txt2Arr(path, type):
    file1 = open(path, 'r')
    Lines = file1.readlines()
    arr1= [];
    for line in Lines:
        if(type == "label"):
            if "<"+type+">" in line: 
                temp = int(line.replace("<"+ type +">", "").replace("</"+type+">", ""))
                arr1.insert(len(arr1),temp)
        if(type == "vector"):
            if "<"+type+">" in line: 
                temp = line.replace("<"+ type +">", "").replace("</"+type+">", "")
                temp1 = temp.split(" ") #string 1 dòng sang array 
                temp2 = [] #array float, 1 vector
                for num in temp1:
                    temp2.insert(len(temp2),float(num))
                arr1.insert(len(arr1),temp2)
    return arr1;






# import numpy as np 

# path0 = 'training-dave.txt'
# path1 = 'testing-dave.txt'


# labelTrain = txt2Arr(path0,"label")
# vectorTrain = txt2Arr(path0,"vector")
# # labelTest = txt2Arr(path1,"label")
# # vectorTest = txt2Arr(path1,"vector")

# vectorTrain = np.array(vectorTrain)

# # print("LABEL TRAIN", labelTrain)
# print("VECTOR TRAIN", vectorTrain)
# # print("LABEL TEST", labelTest)
# # print("VECTOR TEST", vectorTest[0])
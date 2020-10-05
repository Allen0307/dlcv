from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
import math
import sys

pca = PCA(n_components = 4)
trainset = []
testset = []
picture_path = "./p2_data"
for person in range(40):
    for image in range(10):
        im = Image.open(picture_path + "/" + str(person + 1) + "_" + str(image + 1) + ".png")
        data = np.array(im)

        picture = [] #放每張picture的矩陣
        for row in range(56):
            for column in range(46):
                picture.append(data[row][column])
        if image == 9:
            testset.append(picture)
        else:
            trainset.append(picture)
    

trainset = np.array(trainset)
testset = np.array(testset)

print("row : ",np.size(trainset,0))
print("column : ",np.size(trainset,1))
# print("row : ",np.size(testset,0))
# print("column : ",np.size(testset,1))

sum_vector = [0] * 2576

for row in range(360):
    for entry in range(2576):
        sum_vector[entry] += trainset[row][entry]

for entry in range(2576):
    sum_vector[entry] = int(sum_vector[entry] / 360)

def arraytoimage(array):
    eigenface_array = []
    for i in range(2576):
        if i % 46 == 0:
            row = []
            if i != 0:
                eigenface_array.append(row)
        row.append(array[i])
        if i == 2575:
            eigenface_array.append(row)
    return np.array(eigenface_array)

eigenface = Image.fromarray(np.uint8(arraytoimage(sum_vector)))
eigenface.show()

new_data = pca.fit_transform(trainset)
eigenvector = pca.components_
for i in range(len(eigenvector)):
    eigenface = Image.fromarray(np.uint8(arraytoimage(eigenvector[i])))
    eigenface.show()
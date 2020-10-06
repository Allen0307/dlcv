from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
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

#==================================================================
sum_vector = [0] * 2576

for row in range(360):
    for entry in range(2576):
        sum_vector[entry] += trainset[row][entry]
for entry in range(2576):
    sum_vector[entry] = int(sum_vector[entry] / 360)

sum_vector = np.array(sum_vector)
plt.subplot(1, 5, 1)
plt.title("mean")
plt.imshow(sum_vector.reshape(56,46), cmap='gray', vmin=0, vmax=255)

#==================================================================
new_data = pca.fit_transform(trainset)
T_inverse = pca.inverse_transform
eigenvector = pca.components_

for i in range(4):
    plt.subplot(1, 5, i + 2)
    plt.title("i = " + str(i + 1))
    plt.imshow(eigenvector[i].reshape(56,46), cmap='gray')
plt.show()

answer = [0] * 2576
for i in range(360):
    answer += eigenvector[0] * new_data[i][0]


# for i in range(len(eigenvector)):
#     eigenface = Image.fromarray(np.uint8(arraytoimage(eigenvector[i])))
#     eigenface.show()
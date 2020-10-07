from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import math
import sys


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
#mean eigenface

# sum_vector = [0] * 2576

# for row in range(360):
#     for entry in range(2576):
#         sum_vector[entry] += trainset[row][entry]
# for entry in range(2576):
#     sum_vector[entry] = int(sum_vector[entry] / 360)

# sum_vector = np.array(sum_vector)
# plt.subplot(1, 5, 1)
# plt.title("mean")
# plt.imshow(sum_vector.reshape(56,46), cmap='gray', vmin=0, vmax=255)

#==================================================================
#the first 4 eigenface

# pca = PCA(n_components = 4)
# person_eigenvalue = pca.fit_transform(trainset)
# eigenvector = pca.components_

# for i in range(4):
#     plt.subplot(1, 5, i + 2)
#     plt.title("i = " + str(i + 1))
#     plt.imshow(eigenvector[i].reshape(56,46), cmap='gray')
# plt.show()
#==================================================================

#2-1
im = Image.open(picture_path + "/" + '2' + "_" + '1' + ".png")
data = np.array(im)

plt.subplot(1, 6, 1)
plt.imshow(data, cmap = 'gray')
plt.title('Original')

n = [3, 50, 170, 240, 345]
for index in range(len(n)):
    pca = PCA(n_components = n[index])
    person_eigenvalue = pca.fit_transform(trainset)
    eigenvector = pca.components_
    answer = np.zeros((1, 2576))

    eigenface = pca.inverse_transform(person_eigenvalue)

    plt.subplot(1, 6, index + 2)
    plt.title('n = ' + str(n[index]) + '\n' + ' MSE = ' + str(int(mean_squared_error(data,eigenface[9].reshape(56,46)))))
    plt.imshow(eigenface[9].reshape(56,46), cmap = 'gray')
plt.show()
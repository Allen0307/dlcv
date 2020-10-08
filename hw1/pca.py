from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
import math
import sys


trainset = []
testset = []
picture_path = "./p2_data"
for image in range(10):
    for person in range(40):
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
train_label = []
test_label = []
for i in range(9):
    for j in range(40):
        train_label.append(j + 1)
for k in range(40):
    test_label.append(i + 1)

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
# im = Image.open(picture_path + "/" + '2' + "_" + '1' + ".png")
# data = np.array(im)

# plt.subplot(1, 6, 1)
# plt.imshow(data, cmap = 'gray')
# plt.title('Original')

# n = [3, 50, 170, 240, 345]
# for index in range(len(n)):
#     pca = PCA(n_components = n[index])
#     person_eigenvalue = pca.fit_transform(trainset)
#     eigenvector = pca.components_
#     answer = np.zeros((1, 2576))

#     eigenface = pca.inverse_transform(person_eigenvalue)

#     plt.subplot(1, 6, index + 2)
#     plt.title('n = ' + str(n[index]) + '\n' + ' MSE = ' + str(int(mean_squared_error(data,eigenface[9].reshape(56,46)))))
#     plt.imshow(eigenface[9].reshape(56,46), cmap = 'gray')
# plt.show()

#==================================================================
#2-4
f = open('./2-problem/2-4', 'w')
n = [3, 50, 170]
k = [1, 3, 5]

label_1 = train_label[0:240]
label_2 = train_label[120:]
label_3 = train_label[240:] + train_label[0:120]

def model(n, k):
    pca = PCA(n_components = n)
    knn = KNeighborsClassifier(n_neighbors = k)
    person_eigenvalue = pca.fit_transform(trainset)
    pca_trainset = pca.inverse_transform(person_eigenvalue)
    trainset_1 = np.concatenate((pca_trainset[0:120], pca_trainset[120:240]))
    trainset_2 = np.concatenate((pca_trainset[120:240], pca_trainset[240:]))
    trainset_3 = np.concatenate((pca_trainset[240:], pca_trainset[0:120]))
    train = [trainset_1, trainset_2, trainset_3]
    label = [label_1, label_2, label_3]
    position = [2, 0, 1]
    best_acc = 0
    best_fold = 0
    for i in range(3):
        knn.fit(train[i], label[i])
        predict = knn.predict(train[position[i]])
        correct = 0  
        total = 0
        for index in range(len(predict)):
            if predict[index] == label[i][index]:
                correct += 1
            total += 1
        if correct / total >= best_acc:
            best_acc = correct / total
            best_fold = i
    f.write("n = " + str(n) + ", k = " + str(k) + ", fold = " + str(best_fold) + ", acc = " + str(best_acc) + '\n')

for i in k:
    for j in n:
        model(j, i)
#==================================================================
from PIL import Image
import numpy as np
import pandas as pd
import math

k_SIZE = 32
im = Image.open("./bird.jpg")
data = np.array(im)

compute = []

for row in range(1024):
    for column in range(1024):
        node = {}
        node['row'] = row
        node['column'] = column
        node['r'] = data[row][column][0]
        node['g'] = data[row][column][1]
        node['b'] = data[row][column][2]
        compute.append(node)

#initial k's point
np.random.seed(0)
initial = np.random.randint(0, 255, size = (k_SIZE, 3))

#計算每個node的distance,class
def norm2(a, initial):
    #a is a dict. ex: a = {'r':100}
    #b is an array. ex: b = [172 47 117]
    min_distance = 100000
    for i in range(k_SIZE):
        distance_ = 'distance_' + str(i)
        distance = math.sqrt((a['r'] - initial[i][0])**2 + (a['g'] - initial[i][1])**2 + (a['b'] - initial[i][2])**2)
        if(distance < min_distance):
            min_distance = distance
            min_num = i
        a[distance_] = distance
    a['class'] = min_num

for hello in range(3):
    for i in range(len(compute)):
        norm2(compute[i], initial)

    sum_rgb = [] #記錄每個class所有的rgb
    for i in range(k_SIZE):
        sum_rgb.append([0,0,0])
    count = [0] * k_SIZE #記錄每個class有多少個

    for i in range(len(compute)):
        sum_rgb[compute[i]['class']][0] += compute[i]['r']
        sum_rgb[compute[i]['class']][1] += compute[i]['g']
        sum_rgb[compute[i]['class']][2] += compute[i]['b']
        count[compute[i]['class']] += 1
    
    for i in range(k_SIZE):
        if count[i] != 0:
            for j in range(3):
                sum_rgb[i][j] = int(sum_rgb[i][j] / count[i])
        else:
            sum_rgb[i] = initial[i]
    initial = sum_rgb

answer = []
index = 0
def combine(a):
    return initial[a['class']]

for i in range(1024):
    row_init = []
    for j in range(1024):
        row_init.append(combine(compute[index]))
        index += 1
    answer.append(row_init)

answer = np.array(answer)
new_im = Image.fromarray(np.uint8(answer))
new_im.show()



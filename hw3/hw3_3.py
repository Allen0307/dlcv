import csv

label = []

with open('hw3_data/digits/mnistm/test.csv',newline='') as csvfile:
    rows = csv.reader(csvfile)
    count = 0
    for row in rows:
        if count == 0:
            count+=1
            continue
        else:
            label.append(row[1])

my_label = []


with open('test/p4_M.csv',newline='') as csvfile:
    rows = csv.reader(csvfile)
    count = 0
    for row in rows:
        if count == 0:
            count+=1
            continue
        else:
            my_label.append(row[1])

correct = 0
total = 0
for i in range(len(label)):
    if label[i] == my_label[i]:
        correct+=1
    total+=1
print('正確率 : ',correct/total)
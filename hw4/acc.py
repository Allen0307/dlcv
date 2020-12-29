import csv
import sys
val_dic = {}
with open('hw4_data/val.csv',newline='') as csvfile:
    rows = csv.reader(csvfile)
    title = True
    for row in rows:
        if title == True:
            title = False
            continue
        else:
            val_dic[row[0]] = row[2]
print(val_dic)
total_ans = []
with open('hw4_data/val_5_shot.csv', newline='') as csvfile:
    rows = csv.reader(csvfile)
    title = True
    for row in rows:
        if title == True:
            title = False
            continue
        else:
            row = row[2:]
            label = {}

            label[val_dic[row[4]]] = 0 #label1
            label[val_dic[row[9]]] = 1 #label1 
            label[val_dic[row[14]]]= 2#label1 
            label[val_dic[row[19]]] = 3#label1 
            label[val_dic[row[24]]] = 4#label1     
            row = row[25:]       

            
            temp_ans = []
            for i in range(len(row)):
                temp_ans.append(label[val_dic[row[i]]])
            total_ans.append(temp_ans)
print(total_ans)
my_ans = []
with open('test.csv', newline='') as csvfile:
    rows = csv.reader(csvfile)
    title = True
    for row in rows:
        if title == True:
            title = False
            continue
        else:
            row = row[1:]
            temp_acc = []
            for i in range(len(row)):
                temp_acc.append(row[i])
        my_ans.append(temp_acc)

print(len(total_ans[0]))
print(len(my_ans[0]))
correct = 0
total = 0
for i in range(400):
    for j in range(75):

        if int(my_ans[i][j]) == int(total_ans[i][j]):
            correct +=1
        total+=1

print('acc = ', correct/total)
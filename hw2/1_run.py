import csv
import sys
readfile_path = 'test/test_pred.csv'

with open(readfile_path, newline='') as csvfile:

  # 讀取 CSV 檔案內容
    rows = csv.reader(csvfile)

  # 以迴圈輸出每一列
    correct = 0
    total = 0
    for row in rows:
        total += 1
        if row[0] == 'image_id':
            total -= 1
            continue
        myans = row[1]
        if row[0][1] == '_':
            label = row[0][0]

        else:
            label = row[0][0:2]

        if(label == myans):
            correct += 1
print('全部資料 : ', total)
print('我答對了 : ', correct)
print('答對率 : ', correct/total)

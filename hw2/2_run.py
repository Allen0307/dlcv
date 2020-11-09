from mean_iou_evaluate import read_masks, mean_iou_score

myans_path = 'improve/'
label_path = 'hw2_data/p2_data/validation/'

myans = read_masks(myans_path)
label = read_masks(label_path)

score = mean_iou_score(myans, label)
print(score)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import imageio
import matplotlib.pyplot as plt
from PIL import Image
import scipy.misc
import argparse

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

def mean_iou_score(pred, labels):
    '''
    Compute mean IoU score over 6 classes
    '''
    mean_iou = 0
    for i in range(6):
        tp_fp = np.sum(pred == i)
        tp_fn = np.sum(labels == i)
        tp = np.sum((pred == i) * (labels == i))
        iou = tp / (tp_fp + tp_fn - tp)
        mean_iou += iou / 6
        print('class #%d : %1.5f'%(i, iou))
    print('\nmean_iou: %f\n' % mean_iou)

    return mean_iou

def read_masks(filepath):
    '''
    Read masks from directory and tranform to categorical
    '''
    file_list = [file for file in os.listdir(filepath) if file.endswith('.png')]
    file_list.sort()
    print(file_list)
    n_masks = len(file_list)
    masks = np.empty((n_masks, 512, 512))

    for i, file in enumerate(file_list):

        mask = imageio.imread(os.path.join(filepath, file))
        mask = (mask >= 128).astype(int)
        mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
        masks[i, mask == 3] = 0  # (Cyan: 011) Urban land 
        masks[i, mask == 6] = 1  # (Yellow: 110) Agriculture land 
        masks[i, mask == 5] = 2  # (Purple: 101) Rangeland 
        masks[i, mask == 2] = 3  # (Green: 010) Forest land 
        masks[i, mask == 1] = 4  # (Blue: 001) Water 
        masks[i, mask == 7] = 5  # (White: 111) Barren land 
        masks[i, mask == 0] = 6  # (Black: 000) Unknown
        masks[i, mask == 4] = 6  # (Red: 100) Unknown

    return masks

mask_path_train = 'hw2_data/p2_data/train/'
mask_path_val = 'hw2_data/p2_data/validation/'
label_train = read_masks(mask_path_train)
label_val = read_masks(mask_path_val)

cls_color = {
    0:  [0, 255, 255],
    1:  [255, 255, 0],
    2:  [255, 0, 255],
    3:  [0, 255, 0],
    4:  [0, 0, 255],
    5:  [255, 255, 255],
    6: [0, 0, 0],
}
class Allen(Dataset): 
    def __init__(self, mode, transform = None):
        self.mode = mode
        self.transform = transform

        if self.mode == 'train':
            self.common_path = 'hw2_data/p2_data/train/'
            self.filename = [file for file in os.listdir(self.common_path) if file.endswith('.jpg')]
            self.filename.sort()

        else:
            self.common_path = 'hw2_data/p2_data/validation/'
            self.filename = [file for file in os.listdir(self.common_path) if file.endswith('.jpg')]
            self.filename.sort()

        self.len = len(self.filename)
        print(self.len)

    def __getitem__(self, index):

        sat_path = os.path.join(self.common_path + str(self.filename[index]))
        sat_im = Image.open(sat_path)

        if self.transform is not None:
            sat_im = self.transform(sat_im)

        if self.mode == 'train':
            label = label_train[index]
        else:
            label = label_val[index]

        return sat_im, torch.tensor(label, dtype=torch.long), str(self.filename[index][0:4])

    def __len__(self):

        return(self.len)

trainset = Allen('train', transform = transforms.ToTensor())
valset = Allen('val', transform = transforms.ToTensor())
trainset_loader = DataLoader(trainset, batch_size = 8, shuffle = True)
valset_loader = DataLoader(valset, batch_size = 8, shuffle = False)
print('訓練資料大小 : ',trainset.__len__())
print('評估資料大小 : ',valset.__len__())

def visual(A):
  #將512*512的nparray 轉成512*512*3
  A = list(A.reshape(512*512))

  for index in range(512*512):

      entry = np.array(cls_color[int(A[index])])

      A[index] = entry

  A = np.array(A)
  A = A.reshape((512,512,3))
  return A

class Net(nn.Module):
    def __init__(self, model):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(*list(model.children())[:-2][0])
        self.fcn = nn.Sequential(
            nn.Conv2d(512, 4096, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(4096,4096,1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(4096,7,1),
            nn.ReLU(),
            nn.ConvTranspose2d(7,7,64,32,16)
        )


    def forward(self, x):
      x = self.layer1(x)
      x = self.fcn(x)
      return x


model = torchvision.models.vgg16(pretrained = True)
mymodel = Net(model).to(device)
print(mymodel)

def val(model,epoch):
    model.eval()
    with torch.no_grad():

      for batch_id, data in enumerate(valset_loader):
          im, label, name = data
          im, label = im.to(device), label.to(device)
          output = model(im)
          pred = output.max(1, keepdim = True)[1]
          pred = pred.cpu().numpy().reshape((np.size(pred,0),512,512))

          #=========================================================================
          if epoch == 0:
              for i in range(len(name)):
                  if name[i] == '0010':
                      hello = pred[i]
                      hello = hello.reshape((512, 512))
                      hello = visual(hello)
                      plt.subplot(3,3,1)
                      plt.title('epoch = ' + str(epoch) + ', id = ' + str(name[i]))
                      plt.imshow(hello)
                  elif name[i] == '0097':
                      hello = pred[i]
                      hello = hello.reshape((512, 512))
                      hello = visual(hello)
                      plt.subplot(3,3,2)
                      plt.title('epoch = ' + str(epoch) + ', id = ' + str(name[i]))
                      plt.imshow(hello)
                  elif name[i] == '0107':
                      hello = pred[i]
                      hello = hello.reshape((512, 512))
                      hello = visual(hello)
                      plt.subplot(3,3,3)
                      plt.title('epoch = ' + str(epoch) + ', id = ' + str(name[i]))
                      plt.imshow(hello)
          elif epoch == 9:
              for i in range(len(name)):
                  if name[i] == '0010':
                      hello = pred[i]
                      hello = hello.reshape((512, 512))
                      hello = visual(hello)
                      plt.subplot(3,3,4)
                      plt.title('epoch = ' + str(epoch) + ', id = ' + str(name[i]))
                      plt.imshow(hello)
                  elif name[i] == '0097':
                      hello = pred[i]
                      hello = hello.reshape((512, 512))
                      hello = visual(hello)
                      plt.subplot(3,3,5)
                      plt.title('epoch = ' + str(epoch) + ', id = ' + str(name[i]))
                      plt.imshow(hello)
                  elif name[i] == '0107':
                      hello = pred[i]
                      hello = hello.reshape((512, 512))
                      hello = visual(hello)
                      plt.subplot(3,3,6)
                      plt.title('epoch = ' + str(epoch) + ', id = ' + str(name[i]))
                      plt.imshow(hello)
          elif epoch == 19:
                for i in range(len(name)):
                    if name[i] == '0010':
                        hello = pred[i]
                        hello = hello.reshape((512, 512))
                        hello = visual(hello)
                        plt.subplot(3,3,7)
                        plt.title('epoch = ' + str(epoch) + ', id = ' + str(name[i]))
                        plt.imshow(hello)
                    elif name[i] == '0097':
                        hello = pred[i]
                        hello = hello.reshape((512, 512))
                        hello = visual(hello)
                        plt.subplot(3,3,8)
                        plt.title('epoch = ' + str(epoch) + ', id = ' + str(name[i]))
                        plt.imshow(hello)
                    elif name[i] == '0107':
                        hello = pred[i]
                        hello = hello.reshape((512, 512))
                        hello = visual(hello)
                        plt.subplot(3,3,9)
                        plt.title('epoch = ' + str(epoch) + ', id = ' + str(name[i]))
                        plt.imshow(hello)                   
          #=========================================================================

          if batch_id == 0:
              ans = pred
          else:
              ans = np.concatenate((ans, pred), axis=0)

      print('\n')
      score = mean_iou_score(ans, label_val)
      return score

def train(model, epoch, time = 100):
    optimizer = optim.Adam(model.parameters(), lr = 0.0001)
    criterion = nn.CrossEntropyLoss()
    model.train()
    iteration = 0 
    best_acc = 0.71
    for ep in range(epoch):
        for id, data in enumerate(trainset_loader):
            im, label, name = data
            im, label = im.to(device), label.to(device)           
            optimizer.zero_grad()
            output = model(im)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            if iteration % time == 0:
                print("train epoch : ",ep + 1," loss : ",loss.item())
            iteration += 1
            
        temp_acc = val(model,ep)

        if temp_acc > best_acc:
            torch.save(model, 'p2_baseline')
            best_acc = temp_acc
            print('good!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!, the best acc = ', best_acc)

if __name__ == '__main__':
    plt.figure(figsize=(10, 10))
    train(mymodel, 20)
    plt.show()
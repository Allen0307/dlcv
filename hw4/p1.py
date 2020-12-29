import os
import sys
import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler

import csv
import random
import numpy as np
import pandas as pd

from PIL import Image
filenameToPILImage = lambda x: Image.open(x)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

class Convnet(nn.Module):
    def __init__(self, in_channels = 3, hid_channels = 64, out_channels = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(in_channels, hid_channels),
            conv_block(hid_channels, hid_channels),
            conv_block(hid_channels, hid_channels),
            conv_block(hid_channels, out_channels),
        )
    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)
def conv_block(in_channels, out_channels):
    bn = nn.BatchNorm2d(out_channels)
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding = 1),
        bn,
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits

# mini-Imagenet dataset
class MiniDataset(Dataset):
    def __init__(self, csv_path, data_dir):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")

        self.transform = transforms.Compose([
            filenameToPILImage,
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        path = self.data_df.loc[index, "filename"]
        label = self.data_df.loc[index, "label"]
        image = self.transform(os.path.join(self.data_dir, path))
        return image, label

    def __len__(self):
        return len(self.data_df)

class GeneratorSampler(Sampler):
    def __init__(self, episode_file_path):
        episode_df = pd.read_csv(episode_file_path).set_index("episode_id")
        self.sampled_sequence = episode_df.values.flatten().tolist()

    def __iter__(self):
        return iter(self.sampled_sequence) 

    def __len__(self):
        return len(self.sampled_sequence)

def predict(model, data_loader):
    prediction_results = []
    with torch.no_grad():
        # each batch represent one episode (support data + query data)
        for i, (data, target) in enumerate(data_loader):

            # split data into support and query data
            support_input = data[:N_way * N_shot,:,:,:].to(device)
            query_input   = data[N_way * N_shot:,:,:,:].to(device)

            # create the relative label (0 ~ N_way-1) for query data
            # label_encoder = {target[i * N_shot] : i for i in range(N_way)}
            # query_label = torch.cuda.LongTensor([label_encoder[class_name] for class_name in target[N_way * N_shot:]])

            # TODO: extract the feature of support and query data
            proto = model(support_input)
            # TODO: calculate the prototype for each class according to its support data
            proto = proto.reshape(N_shot, N_way, -1).mean(dim=0)
            # TODO: classify the query data depending on the its distense with each prototype
            logits = euclidean_metric(model(query_input), proto)
            pred = torch.max(logits, 1)[1]
            prediction_results.append(pred.cpu().numpy())
    return prediction_results

# def parse_args():
#     parser = argparse.ArgumentParser(description="Few shot learning")
#     parser.add_argument('--N-way', default=5, type=int, help='N_way (default: 5)')
#     parser.add_argument('--N-shot', default=1, type=int, help='N_shot (default: 1)')
#     parser.add_argument('--N-query', default=15, type=int, help='N_query (default: 15)')
#     parser.add_argument('--load', type=str, help="Model checkpoint path")
#     parser.add_argument('--test_csv', type=str, help="Testing images csv file")
#     parser.add_argument('--test_data_dir', type=str, help="Testing images directory")
#     parser.add_argument('--testcase_csv', type=str, help="Test case csv")
#     parser.add_argument('--output_csv', type=str, help="Output filename")

#     return parser.parse_args()

if __name__=='__main__':
    # args = parse_args()
    test_csv = sys.argv[1]
    test_data_dir = sys.argv[2]
    testcase_csv = sys.argv[3]
    output_dir = sys.argv[4]
    N_way = 5
    N_shot = 1
    N_query = 15
    test_dataset = MiniDataset(test_csv, test_data_dir)

    test_loader = DataLoader(
        test_dataset, batch_size=N_way * (N_query + N_shot),
        num_workers=3, pin_memory=False, worker_init_fn=worker_init_fn,
        sampler=GeneratorSampler(testcase_csv))

    # TODO: load your model
    model = Convnet().to(device)
    if use_cuda == True:
        model = torch.load('p1_model/p1')
    else:
        model = torch.load('p1_model/p1', map_location=torch.device('cpu'))
    prediction_results = predict(model, test_loader)

    # TODO: output your prediction to csv
    with open(output_dir, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        title = ['episode_id']
        for i in range(75):
            name = 'query'+str(i)
            title.append(name)
        writer.writerow(title)
        for i in range(len(prediction_results)):
            ans = [i]
            for index in range(len(prediction_results[i])):
                ans.append(prediction_results[i][index].item())
            writer.writerow(ans)

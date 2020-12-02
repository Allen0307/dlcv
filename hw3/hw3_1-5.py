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
from PIL import Image
import sys
from torch.autograd import Variable
import matplotlib.pyplot as plt
import argparse
from sklearn.manifold import TSNE
import csv
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

label = []


with open('hw3_data/face/test.csv',newline='') as csvfile:
    rows = csv.reader(csvfile)
    count = 0
    for row in rows:
        if count == 0:
            count+=1
            continue
        else:
            label.append(row[10])

class Allen(Dataset): 
    def __init__(self, transform = None):
        self.transform = transform
        self.common_path = 'hw3_data/face/test/'
        self.filename = sorted(os.listdir(self.common_path))
        self.len = len(self.filename)

    def __getitem__(self, index):
        path = os.path.join(self.common_path + str(self.filename[index]))
        im = Image.open(path)
        if self.transform is not None:
            im = self.transform(im)
        
        return im

    def __len__(self):

        return(self.len)

maleset = Allen(transform = transforms.ToTensor())


male_loader = DataLoader(maleset, batch_size = 1)




class VAE(nn.Module):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims = None) -> None:
        super(VAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return  self.decode(z), mu, log_var, z

model = VAE(in_channels = 3, latent_dim = 512).to(device)

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps * std + mu

if use_cuda == True:
    model = torch.load('p1_model/1117')
else:
    model = torch.load('p1_model/1117', map_location=torch.device('cpu'))

torch.manual_seed(644)



# loss_function = nn.MSELoss()
# model.eval()
# with torch.no_grad():
#   num = 1
#   for batch_id, data in enumerate(test_loader):
#     data = data.to(device)
#     output,a,b = model(data)
#     MSE_loss = loss_function(output, data)
#     image = np.transpose(output[0].cpu(),(1,2,0))
#     image = np.clip(image, 0, 1)

#     # image = trans_rgb(output[0].cpu())
#     data = np.transpose(data[0].cpu(),(1,2,0))


#     if MSE_loss.item()<=0.0038 and num<=10:
#       plt.subplot(2,10,num)
#       plt.title('MSE = '+ str(round(MSE_loss.item(),4)))
#       fig = plt.gcf()
#       fig.set_size_inches(11,4)
#       plt.gca().xaxis.set_major_locator(plt.NullLocator())
#       plt.gca().yaxis.set_major_locator(plt.NullLocator())
#       plt.subplots_adjust(top=1,bottom=0,right=1,left=0,hspace=0,wspace=0)
#       plt.margins(0,0)
#       plt.imshow((image * 255).numpy().astype(np.uint8))

#       plt.subplot(2,10,num+10)
#       fig = plt.gcf()
#       fig.set_size_inches(11,4)
#       plt.gca().xaxis.set_major_locator(plt.NullLocator())
#       plt.gca().yaxis.set_major_locator(plt.NullLocator())
#       plt.subplots_adjust(top=1,bottom=0,right=1,left=0,hspace=0,wspace=0)
#       plt.margins(0,0)
#       plt.imshow(data)
#       num+=1

# plt.show()
model.eval()
with torch.no_grad():
    for batch_idx, data in enumerate(male_loader):
        data = data.to(device)
        a, mu, var, z = model(data)
        # output = reparameterize(mu,var)
        output = z
        if batch_idx == 0:
            male_ans = output
        else:
            male_ans = np.concatenate((male_ans, output), axis=0)   


# print(ans)
print(label)
male_embedded = TSNE(n_components=2).fit_transform(male_ans)
color_map = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
for i in range(len(maleset)):
    if label[i] == '0.0':
        count = 0
    else:
        count = 1
    plt.scatter(male_embedded[i,0],male_embedded[i,1],label = label[i], c=color_map[count])

plt.show()
# plt.show()

# 1117 seed 168
# 1118 seed 69
    
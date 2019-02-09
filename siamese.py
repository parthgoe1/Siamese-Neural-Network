import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from glob import glob
import os
import cv2

torch.__version__

training_dir = "./dataset/train/"
testing_dir = "./dataset/test/"
train_batch_size = 64
train_number_epochs = 100

def imshow(img,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()    


folder_dataset = dset.ImageFolder(root=training_dir)
    
genuine_pairs = []
imposter_pairs = []

#genuine pairs(0)
k = 0
for i in range(0,360,1):
    if(i%10==0 and i!=0):
        k = k + 10
    for j in range(i+1,k+10,1):
        #print(i,j)
        genuine_pairs.append([folder_dataset.imgs[i][0], folder_dataset.imgs[j][0], 0])
              
#imposter pairs(1)
k=10
for i in range(0,360,1):
    for j in range(i+k,360,1):
        imposter_pairs.append([folder_dataset.imgs[i][0], folder_dataset.imgs[j][0], 1])
    k=k-1
    if(k == 0):
        k = 10
        
class SiameseNetworkDataset(Dataset):
    
    def __init__(self,genuine_pairs,imposter_pairs,transform=None,should_invert=True):
        self.genuine_pairs = genuine_pairs
        self.imposter_pairs = imposter_pairs
        self.transform = transform
        self.should_invert = should_invert        
        
    def __getitem__(self,index):
        #choosing randomly betweeen genuine(0) and imposter(1) pair
        choice = random.randint(0,1)
        
        #using if else to choose a random pair
        if(choice == 0):
            #genuine pair(0)
            
            #select a genuine pair
            pair = random.choice(self.genuine_pairs)
            
            #image 1
            img0 = pair[0]
            
            #image 2
            img1 = pair[1]
            
            #genuine label
            label = pair[2]
            
        else:
            #imposter pair(1)
            
            #select an imposter pair
            pair = random.choice(self.imposter_pairs)
            
            #image 1
            img0 = pair[0]
            
            #image 2
            img1 = pair[1]
            
            #genuine label
            label = pair[2]
            
        #open the images
        img0 = Image.open(img0)
        img1 = Image.open(img1)
        
        #converting the images to black and white
        img0 = img0.convert("L")
        img1 = img1.convert("L")
 
        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
            
        return img0, img1, torch.from_numpy(np.array([int(label)], dtype=np.float32))
    
    def __len__(self):
        return len(self.genuine_pairs) + len(self.imposter_pairs)


siamese_dataset = SiameseNetworkDataset(genuine_pairs = genuine_pairs, imposter_pairs = imposter_pairs,
                                        transform=transforms.Compose([transforms.Resize((105,105)),
                                                                      transforms.ToTensor()])                                                                      
                                       ,should_invert=False)
    
# =============================================================================
# vis_dataloader = DataLoader(siamese_dataset,
#                         shuffle=True,
#                         #num_workers=8,
#                         batch_size=8)
# 
# dataiter = iter(vis_dataloader)
# 
# 
# example_batch = next(dataiter)
# concatenated = torch.cat((example_batch[0],example_batch[1]),0)
# imshow(torchvision.utils.make_grid(concatenated))
# print(example_batch[2].numpy())
# 
# =============================================================================

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        
        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=10, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        
        # Max Pooling 1
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=7, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        
        # Max Pooling 2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Convolution 3
        self.cnn3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4, stride=1, padding=0)
        self.relu3 = nn.ReLU()
        
        # Max Pooling 3
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Convolution 4
        self.cnn4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=1, padding=0)
        self.relu4 = nn.ReLU()
        
        # Fully Connected 1
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.sig1 = nn.Sigmoid()
        
        # Fully Connected 2
        self.fc2 = nn.Linear(4096, 1)
        self.sig2 = nn.Sigmoid()
        
    def forward_once(self, x):
        
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)
        
        # Max pool 1
        out = self.pool1(out)
        
        # Convolution 2 
        out = self.cnn2(out)
        out = self.relu2(out)
        
        # Max pool 2
        out = self.pool2(out)
        
        # Convolution 3
        out = self.cnn3(out)
        out = self.relu3(out)
        
        # Max pool 3
        out = self.pool3(out)
        
        # Convolution 4
        out = self.cnn4(out)
        out = self.relu4(out)
        
        # Resize
        out = out.view(out.size(0), -1)
        
        # Fully Connected 1
        out = self.fc1(out)
        out = self.sig1(out)
        
        # Resize
        out = out.view(out.size(0), -1)
        
        # Fully Connected 2
        out = self.fc2(out)
        out = self.sig2(out)
        
        return out
    
    def forward(self, input1, input2):
        
        # forward pass of input 1
        output1 = self.forward_once(input1)
        
        # forward pass of input 2
        output2 = self.forward_once(input2)
        
        return output1, output2
        
        
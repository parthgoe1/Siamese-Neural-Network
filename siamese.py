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
    
vis_dataloader = DataLoader(siamese_dataset,
                        shuffle=True,
                        #num_workers=8,
                        batch_size=8)

dataiter = iter(vis_dataloader)


example_batch = next(dataiter)
concatenated = torch.cat((example_batch[0],example_batch[1]),0)
imshow(torchvision.utils.make_grid(concatenated))
print(example_batch[2].numpy())



        

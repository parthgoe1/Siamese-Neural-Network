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
        

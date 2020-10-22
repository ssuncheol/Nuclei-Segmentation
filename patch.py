import torch 
import torch.nn as nn
import torchvision 
from torchvision.transforms import transforms
import cv2
from sklearn.feature_extraction import image 
import os
import numpy as np
import matplotlib.pyplot as plt 


# stride & patch_size
r = 64
stride = 32
count = 0

#Patch_Image

file_name=os.listdir('/daintlab/data/TNBC/Image')
file_name.sort()


for i in file_name:
 img =cv2.imread(os.path.join('/daintlab/data/TNBC/Image','%s'%i))
 print(img.shape)
 for row in range(15):
    for col in range(15):
      b=img[row*stride:row*stride+r, col*stride:col*stride+ r]
      plt.imsave('/daintlab/data/TNBC/patch_Image/Image%s.png'%count,b)
      count+=1



mask_name=os.listdir('/daintlab/data/TNBC/mask')
mask_name.sort()
    

#Patch_mask


count1 = 0
for i in mask_name:
  mask=cv2.imread(os.path.join('/daintlab/data/TNBC/mask','%s'%i))
  print(mask.shape)
  for row in range(15):
    for col in range(15):
      b=mask[row*stride:row*stride+r, col*stride:col*stride+ r]
      plt.imsave('/daintlab/data/TNBC/patch_mask/Image%s.png'%count1,b)
      count1+=1

#Remove BackGround 

mask_name=os.listdir('/daintlab/data/TNBC/patch_mask')

mask_name.sort()

for i in mask_name:

  mask =cv2.imread(os.path.join('/daintlab/data/TNBC/patch_mask','%s'%i))
  
  mask=mask/.255
  mask[mask>0.5]=1
  mask[mask<0.5]=0 
  
  print(np.sum(mask))
  if np.sum(mask) <200 : 
    print('delete')
    os.remove(os.path.join('/daintlab/data/TNBC/patch_mask','%s'%i))
    os.remove(os.path.join('/daintlab/data/TNBC/patch_Image','%s'%i))
    print('finish')
  else : 
    print('pass')




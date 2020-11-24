import torch 
import torchvision 
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.utils.data.dataset import Dataset
import cv2
import os
from numpy import asarray
import numpy as np
from PIL import Image
import imgaug.augmenters as iaa 
import random







class Nuclei(Dataset): 
    def __init__(self,root,transform=None):
        super(Nuclei,self).__init__()
        self.root = root  
        
        self.image_paths = list(sorted(os.listdir(os.path.join(root,"image"))))
        self.target_paths = list(sorted(os.listdir(os.path.join(root,"mask"))))
        
        self.transform = transform 
          

    def __getitem__(self,index):
        image_name = os.path.join(self.root,"image",self.image_paths[index])
        target_name = os.path.join(self.root,"mask",self.target_paths[index])
        
        
        image = cv2.imread(image_name)
        mask = cv2.imread(target_name)
      
        
        if mask.shape[2] == 4:
            mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
            print(mask.shape)
           
        image =np.array(image)
        mask =np.array(mask)
        

        
        
        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)

        data = {'input': image, 'label': mask}

        return data 

    def __len__(self):         
        return len(self.image_paths)
        



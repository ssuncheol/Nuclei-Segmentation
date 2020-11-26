import cv2
import os
from numpy import asarray
import numpy as np
from PIL import Image
import imgaug.augmenters as iaa 
import random
import torchvision.transforms.functional as tF






class Nuclei(Dataset): 
    def __init__(self,root,transform=None,brightness=0.0,contrast=0.0):
        super(Nuclei,self).__init__()
        self.root = root  
        
        self.image_paths = list(sorted(os.listdir(os.path.join(root,"image"))))
        self.target_paths = list(sorted(os.listdir(os.path.join(root,"mask"))))
        
        self.transform = transform 
        self.brightness = brightness
        self.contrast = contrast   
    
    def __getitem__(self,index):
        image_name = os.path.join(self.root,"image",self.image_paths[index])
        target_name = os.path.join(self.root,"mask",self.target_paths[index])
           
        
        
        image = Image.open(image_name).convert('RGB')
        mask  = Image.open(target_name).convert('L')
        
        
        if self.brightness > 0:
            b = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
            image = tF.adjust_brightness(image,b)
            #mask =  tF.adjust_brightness(mask,b)
        
        if self.contrast > 0:
            c = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
            image = tF.adjust_contrast(image,c)
            #mask =  tF.adjust_contrast(mask,c)
        
        '''
        if mask.shape[2] == 4:
            mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
            print(mask.shape)
        '''   
        image =np.array(image)
        mask =np.array(mask)
        
        mask = mask/255.0
        
    
        
        mask = mask[...,np.newaxis]   
        data = {'input': image, 'label': mask}
        
        if self.transform is not None:
            data = self.transform(data)
         
        return data

    def __len__(self):         
        return len(self.image_paths)



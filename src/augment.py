import torch
import numpy as np 
from PIL import Image
import imgaug.augmenters as iaa
import imgaug as ia
import random
import torchvision.transforms as transforms 
import torchvision.transforms.functional as F

'''
class ToPILImage(object):
    def __call__(self,data): 
        image,mask = data['input'], data['label']
        
         
        
        image = F.to_pil_image(image)
        mask = F.to_pil_image(mask)
        
        data = {'input': image, 'label': mask}
        
        return data
'''   


class ToTensor(object):
    def __call__(self, data):
        image,mask = data['input'], data['label']

        image = image.transpose((2,0,1)).astype(np.float32)
        mask = mask.transpose((2,0,1)).astype(np.float32)

        data = {'input': torch.from_numpy(image), 'label': torch.from_numpy(mask)}
        
        return data


class Normalize(object): 
    
    def __init__(self,mean=0.5,std=0.5): 
        self.mean = mean 
        self.std = std 

    def __call__(self,data): 
        image,mask = data['input'],data['label'] 
        
        image = F.normalize(image,self.mean,self.std)
        #mask = F.normalize(mask,self.mean,self.std)
        
        data = {'input': image, 'label': mask}
        return data

class RandomHorizontalFlip(object): 
    
    def __init__(self,h_alpha):
        self.h_alpha = h_alpha
    
    def __call__(self,data): 
        image,mask = data['input'],data['label']
        
        if np.random.rand() < self.h_alpha : 
            image = np.fliplr(image)
            mask = np.fliplr(mask)
        data = {'input': image, 'label': mask}
        return data 

class RandomVerticalFlip(object): 
    
    def __init__(self,v_alpha):
        self.v_alpha = v_alpha
    
    def __call__(self,data): 
        image,mask = data['input'],data['label']
        
        if np.random.rand() < self.v_alpha : 
            image = np.flipud(image)
            mask = np.flipud(mask)
        data = {'input': image, 'label': mask}
        return data       

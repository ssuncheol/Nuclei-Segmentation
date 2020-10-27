import torch 
import torchvision 
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.utils.data.dataset import Dataset
import cv2
import os
import numpy as np
from PIL import Image




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
      
      image = Image.open(image_name).convert('RGB')
      mask  = Image.open(target_name).convert('L')
      
      image =np.array(image)
      mask = np.array(mask)

      '''
      if np.max(mask)>1:
          mask = mask/.255
          mask[mask>0.5] = 1
          mask[mask<0.5] = 0 
      '''
      if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)

      data = {'input': image, 'label': mask}

      return data 

  def __len__(self):         
      return len(self.image_paths)

transform=transforms.Compose([transforms.ToTensor()])

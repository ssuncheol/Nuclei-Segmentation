from __future__ import print_function
import torch 
import torch.nn as nn
import torch.optim as optim 
import torchvision 
import torchvision.transforms as transforms
import torch.utils.data as data
import matplotlib.pyplot as plt 
import imgaug.augmenters as iaa 
import cv2
import os
import numpy as np
from PIL import Image
import config
import argparse 
import wandb 
from dataloader import Nuclei 
from model import UNET 
from metrics import IOU
import time 
from augment import ToTensor , Normalize ,RandomHorizontalFlip ,RandomVerticalFlip 






def main():
    wandb.init(project="nuclei-segmentation")
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr',
                    type=float,
                    default=0.001,
                    help='learning rate')   
    parser.add_argument('--dropout',
                    type=float,
                    default=0.0,
                    help='dropout rate')
    parser.add_argument('--epochs',
                    type=int,
                    default=100,
                    help='num epochs')
    parser.add_argument('--train_batch',
                    type=int,
                    default=64,
                    help='train batch size')
    parser.add_argument('--val_batch',
                    type=int,
                    default=64,
                    help='validation batch size')
    parser.add_argument('--test_batch',
                    type=int,
                    default=64,
                    help='test batch size')
    parser.add_argument('--brightness',
                    type=float,
                    default=0.0,
                    help='brightness')
    parser.add_argument('--contrast',
                    type=float,
                    default=0.0,
                    help='contrast')
    parser.add_argument('--hflip',
                    type=float,
                    default=0.5,
                    help='hflip')
    parser.add_argument('--vflip',
                    type=float,
                    default=0.5,
                    help='vflip')
    parser.add_argument('--weight_decay',
                    type=float,
                    default=0.001,
                    help='weight_decay')
    parser.add_argument('--gpu',
                    type=str,
                    default='0',
                    help='gpu number')

    args = parser.parse_args()
    wandb.config.update(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    
    #data 
    
    train ='/daintlab/data/TNBC/train'
    val = '/daintlab/data/TNBC/val'
    test = '/daintlab/data/TNBC/test'
    
    
    #transform
    
    transform = transforms.Compose([
                                    #RandomHorizontalFlip(args.hflip),
                                    RandomVerticalFlip(args.vflip),
                                    ToTensor()]
                                    )
    
    def train_load(train,transform):  
        train =Nuclei(train,transform,brightness=args.brightness,contrast=args.contrast)  
        trn_loader = data.DataLoader(train,batch_size=args.train_batch,shuffle=True,num_workers=4)
        return trn_loader 
    
    transform_val = transforms.Compose([ToTensor()])
    val = Nuclei(val,transform_val) 
    val_loader = data.DataLoader(val,batch_size=args.val_batch,shuffle=False,num_workers=4)
    
    #model , criterion , optimizer
     
    model =UNET().cuda()
    
    wandb.watch(model)
    
    criterion = nn.BCEWithLogitsLoss().cuda()
    optimizer = optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)                                
    
    
    #train
    for epoch in range(args.epochs):
        model.train()
        tr_loss=0.0
        t1=time.time()
        trn_loader = train_load(train,transform)
        for batch,nuc in enumerate(trn_loader):
            label = nuc['label'].cuda()
            input = nuc['input'].cuda()
            
            output = model(input)
            optimizer.zero_grad()
            loss = criterion(output,label)
            loss.backward()
            optimizer.step()
            tr_loss += loss 
            del loss 
            del output 
        
            if (batch+1) % 10 == 0 :     
                with torch.no_grad(): 
                    va_loss=0.0
                    iou_values=0.0
                    for j,nuc in enumerate(val_loader):
                        label = nuc['label'].cuda()
                        input = nuc['input'].cuda() 
                        
                        out = model(input)
                        v_loss = criterion(out,label)
                        va_loss += v_loss
    
                        label = label.cpu().numpy()
                        output = out.cpu().numpy()
                        
                        output[output>0.5]=1
                        output[output<0.5]=0
                        
                        iou_value = IOU(output,label)
                        iou_values+=iou_value
                    wandb.log({'IOU_val' : iou_values/len(val_loader)}) 
                       
                print("epoch : {} | step : {} | trn loss : {:.4f} | val loss : {:.4f}".format(epoch,batch+1, tr_loss / 10, va_loss / len(val_loader)))     
                wandb.log({"epoch" : epoch,
                       "trn loss" : tr_loss/10,
                       "val loss" : va_loss/len(val_loader)})
                tr_loss=0.0
                t2=time.time()
                print('train time : {}'.format(t2-t1))
                
                                    
    #test
     
    test = Nuclei(test,transform_val)
    test_loader = data.DataLoader(test,batch_size=args.test_batch,shuffle=True,num_workers=4)
    
    
    raw_images = []
    ground_truth = []
    masking = []
    iou_values=0.0
    
    with torch.no_grad():
        model.eval()
        for batch,nuc in enumerate(test_loader): 
            label = nuc['label'].cuda()
            input = nuc['input'].cuda()
            label = label.cpu().numpy()
            
            out = model(input)
            output = out.cpu().numpy()
            output[output>0.5]=1
            output[output<0.5]=0
            
            raw_images.append(wandb.Image(
            input[0]))
            ground_truth.append(wandb.Image(
            label[0]))
            masking.append(wandb.Image(
            output[0]))
            
            iou_value = IOU(output,label)
            #wandb.log({'IOU of test batch':wandb.Histogram(iou_value)})
            iou_values+=iou_value
            
            wandb.log({'Raw' : raw_images,
                    'Ground truth' : ground_truth,
                    'Masking' : masking})
            wandb.log({'IOU' : iou_values/len(test_loader)})
            #import ipdb; ipdb.set_trace()            
    
    
    
    
    
     
if __name__ == '__main__':
    main()                    

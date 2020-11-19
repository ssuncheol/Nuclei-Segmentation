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
from evaluate import IOU
import time 
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
                    default=50,
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
                    default=32,
                    help='validation batch size')
    parser.add_argument('--weight_decay',
                    type=float,
                    default=0.0001,
                    help='weight_decay')
    parser.add_argument('--gpu',
                    type=str,
                    default='0',
                    help='gpu number')

    args = parser.parse_args()
    wandb.config.update(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    
    #dataloader
    transform=transforms.Compose([
                                  transforms.ToPILImage(),
                                  #transforms.ColorJitter(brightness=0.1),
                                  #transforms.RandomHorizontalFlip(p=0.1),
                                  #transforms.RandomVerticalFlip(p=0.1),
                                  transforms.RandomRotation(degrees=45),
                                  transforms.ToTensor(),
                                  ])
    
    train =Nuclei('/daintlab/data/TNBC/train',transform=transform)
    val = Nuclei('/daintlab/data/TNBC/val',transform=transform)
    test = Nuclei('/daintlab/data/TNBC/test',transform=transform)

    trn_loader = data.DataLoader(train,batch_size=args.train_batch,shuffle=True,num_workers=4)
    val_loader = data.DataLoader(val,batch_size=args.val_batch,shuffle=True,num_workers=4)
    test_loader = data.DataLoader(test,batch_size=args.test_batch,shuffle=False,num_workers=0)
    
    print(len(trn_loader))
    print(len(val_loader))
    print(len(test_loader))
    
    #model 
    
    model =UNET().cuda()
    
    
    #loss function
    
    criterion = nn.BCEWithLogitsLoss().cuda()
    optimizer = optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)

    wandb.watch(model)
    
    #train
    
    for epoch in range(args.epochs):
        model.train()
        tr_loss=0.0
        t1=time.time()
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
                    for j,nuc in enumerate(val_loader): 
                        label = nuc['label'].cuda()
                        input = nuc['input'].cuda()
                        output = model(input)
    
                        v_loss = criterion(output,label)
                        va_loss += v_loss
                print("epoch : {} | step : {} | trn loss : {:.4f} | val loss : {:.4f}".format(epoch,batch+1, tr_loss / 10, va_loss / len(val_loader)))     
                wandb.log({"epoch" : epoch,
                       "trn loss" : tr_loss/10,
                       "val loss" : va_loss/len(val_loader)})
                tr_loss=0.0
                t2=time.time()
                print('train time : {}'.format(t2-t1))
                
    #test 
                
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
            wandb.log({'IOU of test batch':wandb.Histogram(iou_value)})
            iou_values+=iou_value
            
        wandb.log({'Raw' : raw_images,
                   'Ground truth' : ground_truth,
                   'Masking' : masking})
        wandb.log({'IOU' : iou_values/len(test_loader)})
        #import ipdb; ipdb.set_trace()
    
if __name__ == '__main__':
    main()                      

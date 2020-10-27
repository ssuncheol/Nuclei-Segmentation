from __future__ import print_function
import torch 
import torch.nn as nn
import torch.optim as optim 
import torchvision 
import torchvision.transforms as transforms
import torch.utils.data as data
import matplotlib.pyplot as plt 
import cv2
import os
import numpy as np
from PIL import Image
import config
import argparse 
import wandb 


class Nuclei(data.Dataset): 
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
      
      
        if np.max(mask)>1:
            mask = mask/.255
            mask[mask>0.5] = 1
            mask[mask<0.5] = 0 
      
        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)

        nuc = {'input': image, 'label': mask}

        return nuc

    def __len__(self):         
        return len(self.image_paths)


print(model)

class UNET(nn.Module):
    def __init__(self):
        super(UNET,self).__init__()

        def CBR2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=True):    
           layers = []
           layers += [nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,
           stride=stride,padding=padding,bias=bias)]
           layers += [nn.BatchNorm2d(num_features=out_channels)]
           layers += [nn.ReLU()]

           cbr = nn.Sequential(*layers)

           return cbr 

    

        #Encoder 

        self.enc1_1 = CBR2d(3,8)
        self.enc1_2 = CBR2d(8,8)

        self.pool1 = nn.MaxPool2d(kernel_size=2)   

        self.enc2_1 = CBR2d(8,16)
        self.enc2_2 = CBR2d(16,16)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = CBR2d(16,32)
        self.enc3_2 = CBR2d(32,32)

        self.pool3 = nn.MaxPool2d(kernel_size=2)
    
        self.enc4_1 = CBR2d(32,64)
        self.enc4_2 = CBR2d(64,64)

        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.enc5_1 = CBR2d(64,128)

        #Decoder 

        self.dec5_1 = CBR2d(128,64)

        self.unpool4 = nn.ConvTranspose2d(in_channels=64,out_channels=64,kernel_size=2,stride=2,padding=0,bias=True)

        self.dec4_2 = CBR2d(2*64,64)
        self.dec4_1 = CBR2d(64,32)

        self.unpool3 = nn.ConvTranspose2d(in_channels=32,out_channels=32,kernel_size=2,stride=2,padding=0,bias=True)

        self.dec3_2 = CBR2d(2*32,32)
        self.dec3_1 = CBR2d(32,16)

        self.unpool2 = nn.ConvTranspose2d(in_channels=16,out_channels=16,kernel_size=2,stride=2,padding=0,bias=True)

        self.dec2_2 = CBR2d(2*16,16)
        self.dec2_1 = CBR2d(16,8)
 
        self.unpool1 = nn.ConvTranspose2d(in_channels=8,out_channels=8,kernel_size=2,stride=2,padding=0,bias=True)

        self.dec1_2 = CBR2d(2*8,8)
        self.dec1_1 = CBR2d(8,8)

        self.fc = nn.Conv2d(8,1,kernel_size=1,stride=1,padding=0,bias=True)

    def forward(self,x):
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1  = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)


        dec5_1 = self.dec5_1(enc5_1)

        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat((unpool4,enc4_2),dim=1)
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)

        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3,enc3_2),dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2,enc2_2),dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1,enc1_2),dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        x = self.fc(dec1_1)
      
        return x

smooth=1e-6   
def IOU(output,label):
    
    intersection = (output * label).sum(2).sum(1)
    union = (output + label).sum(2).sum(1) 
    
    iou = (intersection+smooth)/(union-intersection+smooth)
    
    return iou
    


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
                    default=1,
                    help='num epochs')
    parser.add_argument('--train_batch',
                    type=int,
                    default=64,
                    help='train batch size')
    parser.add_argument('--val_batch',
                    type=int,
                    default=64,
                    help='validation batch size')
    parser.add_argument('--weight_decay',
                    type=float,
                    default=0.0,
                    help='weight_decay')
    parser.add_argument('--gpu',
                    type=str,
                    default='0',
                    help='gpu number')

    args = parser.parse_args()
    wandb.config.update(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    
    transform=transforms.Compose([transforms.ToTensor()])
    
    train =Nuclei('/daintlab/data/TNBC/train',transform=transform)
    val = Nuclei('/daintlab/data/TNBC/val',transform=transform)
    test = Nuclei('/daintlab/data/TNBC/test',transform=transform)

    trn_loader = data.DataLoader(train,batch_size=args.train_batch,shuffle=True,num_workers=0)
    val_loader = data.DataLoader(val,batch_size=args.val_batch,shuffle=True,num_workers=0)
    test_loader = data.DataLoader(test,batch_size=1,shuffle=False,num_workers=0)
    
    print(len(trn_loader))
    print(len(val_loader))
    print(len(test_loader))
    


    model =UNET().cuda()
    model = nn.DataParallel(model)
    
    criterion = nn.BCEWithLogitsLoss().cuda()
    optimizer = optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)

    wandb.watch(model)
    
    for epoch in range(args.epochs):
        model.train()
        tr_loss=0.0
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
            #label = np.squeeze(label,axis=0)
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
            wandb.log({'IOU':wandb.Table(iou_value)})
            iou_values+=iou_value
        wandb.log({'Raw' : raw_images,
                   'Ground truth' : ground_truth,
                   'Masking' : masking})
        print(iou_values.mean())
        import ipdb; ipdb.set_trace()
            #output = np.squeeze(output,axis=0)
            #output = np.squeeze(output,axis=0)
            #print(output.shape)
            #plt.imsave('/daintlab/data/TNBC/result/result%s.png'%count,output,cmap='gray')
            #count+=1
           
            #wandb.off


if __name__ == '__main__':
    main()                    
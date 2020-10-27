import torch 

smooth=1e-6   
def IOU(output,label):
    
    output=output.squeeze(1)
    label=label.squeeze(1)
    intersection = (output * label).sum(2).sum(1)
    union = (output + label).sum(2).sum(1) 
    
    iou = (intersection+smooth)/(union-intersection+smooth)
    iou = torch.from_numpy(iou)
    thresholded = torch.clamp(20*(iou-0.5),0,10).ceil() / 10
    return thresholded.mean()
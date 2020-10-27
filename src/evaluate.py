smooth=1e-6   
def IOU(output,label):
    
    intersection = (output * label).sum(2).sum(1)
    union = (output + label).sum(2).sum(1) 
    
    iou = (intersection+smooth)/(union-intersection+smooth)
    
    return iou
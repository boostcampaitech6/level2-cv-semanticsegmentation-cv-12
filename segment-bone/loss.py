import torch
import torch.nn.functional as F

def focal_loss(inputs, targets, alpha=.25, gamma=2) : 
    inputs = F.sigmoid(inputs)       
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
    BCE_EXP = torch.exp(-BCE)
    loss = alpha * (1-BCE_EXP)**gamma * BCE
    return loss 

def dice_loss(pred, target, smooth = 1.):
    pred = F.sigmoid(pred)
    pred = pred.contiguous()
    target = target.contiguous()   
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) +   target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()

def IOU_loss(inputs, targets, smooth=1) : 
    inputs = F.sigmoid(inputs)      
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    intersection = (inputs * targets).sum()
    total = (inputs + targets).sum()
    union = total - intersection 
    IoU = (intersection + smooth)/(union + smooth)
    return 1 - IoU

def calc_loss(pred, target):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    iou = IOU_loss(pred, target)
    dice = dice_loss(pred, target)
    focal = focal_loss(pred, target)

    iou_weight = 1.0
    bce_weight = 0.1
    dice_weight = 1.0
    focal_weight = 1.0

    loss = iou * iou_weight + bce * bce_weight + dice * dice_weight + focal * focal_weight
    return loss, iou, bce, dice, focal
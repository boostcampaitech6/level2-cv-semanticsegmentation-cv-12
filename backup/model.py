import torch.nn as nn
from torchvision import models

class FCNResNet50(nn.Module):
    def __init__(self, num_classes):
        super(FCNResNet50, self).__init__()
        
        # Load pre-trained ResNet-50 model
        self.resnet50 = models.segmentation.fcn_resnet50(pretrained=True)
        
        # Replace the classifier to adjust for the number of output classes
        self.resnet50.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)
    
    def forward(self, x):
        return self.resnet50(x)

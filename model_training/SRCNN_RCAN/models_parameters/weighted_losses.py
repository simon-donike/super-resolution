import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.autograd import Variable
from torch.nn.functional import normalize


from torchvision import models
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor




import os
import sys
import math
import torch
import numpy as np
import cv2
    

   
    
class weighted_loss():
    
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # loss weights
        self.pixel_weight = 1.0
        self.content_weight = 1.0
        self.adversarial_weight = 0.001
        
        #self.psnr_criterion = nn.MSELoss().to(self.device) # why psnr MSE?
        self.pixel_criterion = nn.MSELoss().to(self.device)
        self.content_criterion = ContentLoss().to(self.device)
        #self.adversarial_criterion = nn.BCEWithLogitsLoss().to(self.device)


        
    def get_loss(self,sr,hr):
        pixel_loss = self.pixel_weight * self.pixel_criterion(sr, hr.detach()) # detaching hr from generator
        content_loss = self.content_weight * self.content_criterion(sr, hr.detach())
        #adversarial_loss = self.adversarial_weight * self.adversarial_criterion(output, real_label)
        
        # Count discriminator total loss
        g_loss = pixel_loss + content_loss# + adversarial_loss
        return(g_loss)
        
    
class ContentLoss(nn.Module):
    def __init__(self) -> None:
        super(ContentLoss, self).__init__()
        # Load the VGG19 model trained on the ImageNet dataset.
        vgg19 = models.vgg19(pretrained=True).eval()
        # Extract the thirty-sixth layer output in the VGG19 model as the content loss.
        self.feature_extractor = nn.Sequential(*list(vgg19.features.children())[:36])
        # Freeze model parameters.
        for parameters in self.feature_extractor.parameters():
            parameters.requires_grad = False

        # The preprocessing method of the input data. This is the VGG model preprocessing method of the ImageNet dataset.
        # whats that?
        self.register_buffer("mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        # Standardized operations
        # why, whats that?
        sr = sr.sub(self.mean).div(self.std)
        hr = hr.sub(self.mean).div(self.std)

        # Find the feature map difference between the two images
        loss = F.l1_loss(self.feature_extractor(sr), self.feature_extractor(hr))

        return loss
    

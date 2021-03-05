# Required libraries

import torch
import torchvision
import os
import sys
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class MobileNetv2(torch.nn.Module):
    """
    MobileNetv2 pretrained model with additional projection head for transfer learning
    params: out_features - No. of output classes
        
    """
    def __init__(self, out_features):
        
        self.out_features = out_features
        
        super(MobileNetv2, self).__init__()
        
        self.mnet = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
        self.features = self.mnet.features
        
        for params in self.mnet.parameters():
            params.requires_grad = False
            
        self.head = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(62720,512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, self.out_features))
       
        
    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x

    

class SegNetDecoder(torch.nn.Module):
    """
    Implementation of SegNet decoder
    params: in_channels  - input channel size obtained from feature extractor
            out_channels - output channel size of the decoder 
    """
    
    def __init__(self, in_channels, out_channels):
        super(SegNetDecoder, self).__init__()
        
        self.out_channels = out_channels
        self.in_channels = in_channels
        
        self.decoder_convtr_42 = nn.Sequential(*[
                           nn.ConvTranspose2d(in_channels=self.in_channels,out_channels=512, stride=2, kernel_size=3,padding=1),
                                    nn.BatchNorm2d(512)
                                                ])
        
        self.decoder_convtr_41 = nn.Sequential(*[
                                    nn.ConvTranspose2d(in_channels=512,out_channels=512, stride=2, kernel_size=3,padding=1),
                                    nn.BatchNorm2d(512)
                                                ])
        
        self.decoder_convtr_40 = nn.Sequential(*[
                                    nn.ConvTranspose2d(in_channels=512,out_channels=512, stride=2, kernel_size=3,padding=1),
                                    nn.BatchNorm2d(512)
                                                ])
        
        self.decoder_convtr_32 = nn.Sequential(*[
                                    nn.ConvTranspose2d(in_channels=512,out_channels=512,stride=2, kernel_size=3,padding=1),
                                    nn.BatchNorm2d(512)
                                                ])
        
        self.decoder_convtr_31 = nn.Sequential(*[
                                    nn.ConvTranspose2d(in_channels=512,out_channels=512, stride=2, kernel_size=3,padding=1),
                                    nn.BatchNorm2d(512)
                                                ])
        
        self.decoder_convtr_30 = nn.Sequential(*[
                                    nn.ConvTranspose2d(in_channels=512,out_channels=256, stride=1, kernel_size=3,padding=1),
                                    nn.BatchNorm2d(256)
                                                ])
        
        self.decoder_convtr_22 = nn.Sequential(*[
                                    nn.ConvTranspose2d(in_channels=256,out_channels=256,stride=1, kernel_size=3,padding=1),
                                    nn.BatchNorm2d(256)
                                                ])
        
        self.decoder_convtr_21 = nn.Sequential(*[
                                    nn.ConvTranspose2d(in_channels=256,out_channels=256, stride=1,kernel_size=3,padding=1),
                                    nn.BatchNorm2d(256)
                                                ])
        
        self.decoder_convtr_20 = nn.Sequential(*[
                                    nn.ConvTranspose2d(in_channels=256,out_channels=128,stride=1, kernel_size=3,padding=1),
                                    nn.BatchNorm2d(128)
                                                ])
        
        self.decoder_convtr_11 = nn.Sequential(*[
                                    nn.ConvTranspose2d(in_channels=128,out_channels=128,stride=1, kernel_size=3,padding=1),
                                    nn.BatchNorm2d(128)
                                                ])
        
        self.decoder_convtr_10 = nn.Sequential(*[
                                    nn.ConvTranspose2d(in_channels=128,out_channels=64,stride=2, kernel_size=3,padding=1),
                                    nn.BatchNorm2d(64)
                                                ])
        
        self.decoder_convtr_01 = nn.Sequential(*[
                                    nn.ConvTranspose2d(in_channels=64,out_channels=64,stride=2, kernel_size=3,padding=1),
                                    nn.BatchNorm2d(64)
                                                ])
        
        self.decoder_convtr_00 = nn.Sequential(*[
                                    nn.ConvTranspose2d(in_channels=64, out_channels=self.out_channels,kernel_size=3,
                                                                   padding=1)
                                               ])
        
    def forward(self, x):
        
        # Decoder Stage - 5
        x5 = F.relu(self.decoder_convtr_42(x))
        x5 = F.relu(self.decoder_convtr_41(x5))
        x5 = F.relu(self.decoder_convtr_40(x5))
       

        # Decoder Stage - 4
        x4 = F.relu(self.decoder_convtr_32(x5))
        x4 = F.relu(self.decoder_convtr_31(x4))
        x4 = F.relu(self.decoder_convtr_30(x4))
    

        # Decoder Stage - 3
        x3 = F.relu(self.decoder_convtr_22(x4))
        x3 = F.relu(self.decoder_convtr_21(x3))
        x3 = F.relu(self.decoder_convtr_20(x3))

        # Decoder Stage - 2
        x2 = F.relu(self.decoder_convtr_11(x3))
        x2 = F.relu(self.decoder_convtr_10(x2))
   
        # Decoder Stage - 1
        x1 = F.relu(self.decoder_convtr_01(x2))
        x1 = self.decoder_convtr_00(x1)

        return x1
    
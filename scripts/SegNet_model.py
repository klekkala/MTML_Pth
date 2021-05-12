# Required libraries
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torchsummary import summary

#------------------------------------------------------------------------------------------------------------------------------

def _add_decoder_blocks(input_dim, output_dim, name, large_decoder):
    """
    This function returns a block of conv + batch norm + relu layers as a sequential unit with unique names assigned
    :params: input_dim: input channels
             output_dim: output channels
             name: base name of layer
             large_decoder: if True then decoder has 3 deconv layers else two
    """
 
    if large_decoder:
        return nn.Sequential(
            OrderedDict(
                [
                    ( 
                        name + "deconv1", nn.ConvTranspose2d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, padding=1),
                    ),
                    (  name + "norm1", nn.BatchNorm2d(input_dim),),
                    (  name + "act1", nn.ReLU(inplace=True),),

                    ( 
                        name + "deconv2", nn.ConvTranspose2d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, padding=1),
                    ),
                    (  name + "norm2", nn.BatchNorm2d(input_dim)),
                    (  name + "act2", nn.ReLU(inplace=True)),

                    ( 
                        name + "deconv3", nn.ConvTranspose2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, padding=1),
                    ),
                    (  name + "norm3", nn.BatchNorm2d(output_dim)),
                    (  name + "act3", nn.ReLU(inplace=True)),
                ]
            
            )
        )
    else:

        return nn.Sequential(
            OrderedDict(
                [
                    ( 
                        name + "deconv1", nn.ConvTranspose2d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, padding=1),
                    ),
                    (  name + "norm1", nn.BatchNorm2d(input_dim)),
                    (  name + "act1", nn.ReLU(inplace=True)),

                    ( 
                        name + "deconv2", nn.ConvTranspose2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, padding=1),
                    ),
                    (  name + "norm2", nn.BatchNorm2d(output_dim)),
                    (  name + "act2", nn.ReLU(inplace=True)),
                ]
            )
        )


class Encoder(nn.Module):
    """
    Class for the SegNet encoder architecture
    """
    def __init__(self, input_channels, *args, **kwargs):
        super(Encoder, self).__init__(*args, **kwargs)

        self.input_channels = input_channels

        vgg16_bn = torchvision.models.vgg16_bn(pretrained=True)
        maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False, return_indices=True)
        first_layer = nn.Conv2d(in_channels=self.input_channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        features = list(vgg16_bn.features.children())
        weight = features[0].weight.data 

        # We need the maxpooled indices and the original pretrained model doesn't provide it, hence replacing it.
        features[0], features[6], features[13], features[23], features[33], features[43] = first_layer, maxpool, maxpool, maxpool, maxpool, maxpool
        features[0].weight.data = weight

        # encoder
        self.enc1 = nn.Sequential(*features[0:7])
        self.enc2 = nn.Sequential(*features[7:14])
        self.enc3 = nn.Sequential(*features[14:24])
        self.enc4 = nn.Sequential(*features[24:34])
        self.enc5 = nn.Sequential(*features[34:])


class Depth_Decoder(nn.Module):
    """
    Class for Depth Decoder architecture
    """
    def __init__(self, *args, **kwargs):
        super(Depth_Decoder, self).__init__(*args, **kwargs)

        self.depth_dec5 = _add_decoder_blocks(512, 512, "depth_dec5_", True)
        self.depth_dec4 = _add_decoder_blocks(512, 256, "depth_dec4_", True)
        self.depth_dec3 = _add_decoder_blocks(256, 128, "depth_dec3_", True)
        self.depth_dec2 = _add_decoder_blocks(128, 64, "depth_dec2_", False)
        self.depth_dec1 = _add_decoder_blocks(64, 1, "depth_dec1_", False)
        self.depth_dec1.depth_dec1_act2 = nn.Sigmoid()


class Decoder(nn.Module):
    """
    Class implementation of Generic task Decoder
    """
    def __init__(self, task, output_channels, *args, **kwargs):
        super(Decoder, self).__init__(*args, **kwargs)

        self.task = task
        self.output_channels = output_channels

        self.dec5 = _add_decoder_blocks(512, 512, self.task+"_dec5_", True)
        self.dec4 = _add_decoder_blocks(512, 256, self.task+"_dec4_", True)
        self.dec3 = _add_decoder_blocks(256, 128, self.task+"_dec3_", True)
        self.dec2 = _add_decoder_blocks(128, 64, self.task+"_dec2_", False)
        self.dec1 = _add_decoder_blocks(64, self.output_channels, self.task+"_dec1_", False)


class Segmentation(Decoder, Encoder):
    """
    Class implementation of Segmentation Single-task
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, image):
        
        # task encoder
        x, ind1 = self.enc1(image)
        x, ind2 = self.enc2(x)
        x, ind3 = self.enc3(x)
        x, ind4 = self.enc4(x)
        x, ind5 = self.enc5(x)

        # task decoder
        y = nn.MaxUnpool2d(kernel_size=2, stride=2 )(x, ind5) 
        y = self.dec5(y)
        y = nn.MaxUnpool2d(kernel_size=2, stride=2 )(y, ind4) 
        y = self.dec4(y)
        y = nn.MaxUnpool2d(kernel_size=2, stride=2 )(y, ind3) 
        y = self.dec3(y)
        y = nn.MaxUnpool2d(kernel_size=2, stride=2 )(y, ind2) 
        y = self.dec2(y)
        y = nn.MaxUnpool2d(kernel_size=2, stride=2 )(y, ind1) 
        y = self.dec1(y)
        
        return y


class Segmentation_Depth(Depth_Decoder, Segmentation):
    """
    Class implementation of Segmentation Depth Multi-task
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, image):
        
        task_op = super().forward(image)

        x, ind1 = self.enc1(image)
        x, ind2 = self.enc2(x)
        x, ind3 = self.enc3(x)
        x, ind4 = self.enc4(x)
        x, ind5 = self.enc5(x)

        z = nn.MaxUnpool2d(kernel_size=2, stride=2 )(x, ind5) 
        z = self.depth_dec5(z)
        z = nn.MaxUnpool2d(kernel_size=2, stride=2 )(z, ind4) 
        z = self.depth_dec4(z)
        z = nn.MaxUnpool2d(kernel_size=2, stride=2 )(z, ind3) 
        z = self.depth_dec3(z)
        z = nn.MaxUnpool2d(kernel_size=2, stride=2 )(z, ind2) 
        z = self.depth_dec2(z)
        z = nn.MaxUnpool2d(kernel_size=2, stride=2 )(z, ind1) 
        depth_op = self.depth_dec1(z)

        return task_op, depth_op


class Surfac_Normal(Decoder, Encoder):
    """
    Class implementation of Surface Normal Single-task
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dec1.sn_dec1_act2 = nn.Tanh()

    def forward(self, image):
        
        # task encoder
        x, ind1 = self.enc1(image)
        x, ind2 = self.enc2(x)
        x, ind3 = self.enc3(x)
        x, ind4 = self.enc4(x)
        x, ind5 = self.enc5(x)

        # task decoder
        y = nn.MaxUnpool2d(kernel_size=2, stride=2 )(x, ind5) 
        y = self.dec5(y)
        y = nn.MaxUnpool2d(kernel_size=2, stride=2 )(y, ind4) 
        y = self.dec4(y)
        y = nn.MaxUnpool2d(kernel_size=2, stride=2 )(y, ind3) 
        y = self.dec3(y)
        y = nn.MaxUnpool2d(kernel_size=2, stride=2 )(y, ind2) 
        y = self.dec2(y)
        y = nn.MaxUnpool2d(kernel_size=2, stride=2 )(y, ind1) 
        y = self.dec1(y)
        
        return y


class Surface_Normal_Depth(Depth_Decoder, Surfac_Normal):
    """
    Class implementation of Surface Normal Depth Multi-task
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
       
    def forward(self, image):
        
        task_op = super().forward(image)

        x, ind1 = self.enc1(image)
        x, ind2 = self.enc2(x)
        x, ind3 = self.enc3(x)
        x, ind4 = self.enc4(x)
        x, ind5 = self.enc5(x)

        z = nn.MaxUnpool2d(kernel_size=2, stride=2 )(x, ind5) 
        z = self.depth_dec5(z)
        z = nn.MaxUnpool2d(kernel_size=2, stride=2 )(z, ind4) 
        z = self.depth_dec4(z)
        z = nn.MaxUnpool2d(kernel_size=2, stride=2 )(z, ind3) 
        z = self.depth_dec3(z)
        z = nn.MaxUnpool2d(kernel_size=2, stride=2 )(z, ind2) 
        z = self.depth_dec2(z)
        z = nn.MaxUnpool2d(kernel_size=2, stride=2 )(z, ind1) 
        depth_op = self.depth_dec1(z)

        return task_op, depth_op



class VP(Encoder):
    """
    Class for Vanishing point Model architecture.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.head = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(55296, 2048),
                    nn.BatchNorm1d(2048),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(2048, 1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(1024, 9),
                    nn.BatchNorm1d(9),
                )

    def forward(self, image):
        
        # task encoder
        x, ind1 = self.enc1(image)
        x, ind2 = self.enc2(x)
        x, ind3 = self.enc3(x)
        x, ind4 = self.enc4(x)
        x, ind5 = self.enc5(x)

        y = self.head(x)
        
        return y


class VP_Depth(Depth_Decoder, VP):
    """
    Class implementation of Vp Depth Multi-task
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
       
    def forward(self, image):
        
        task_op = super().forward(image)

        x, ind1 = self.enc1(image)
        x, ind2 = self.enc2(x)
        x, ind3 = self.enc3(x)
        x, ind4 = self.enc4(x)
        x, ind5 = self.enc5(x)

        z = nn.MaxUnpool2d(kernel_size=2, stride=2 )(x, ind5) 
        z = self.depth_dec5(z)
        z = nn.MaxUnpool2d(kernel_size=2, stride=2 )(z, ind4) 
        z = self.depth_dec4(z)
        z = nn.MaxUnpool2d(kernel_size=2, stride=2 )(z, ind3) 
        z = self.depth_dec3(z)
        z = nn.MaxUnpool2d(kernel_size=2, stride=2 )(z, ind2) 
        z = self.depth_dec2(z)
        z = nn.MaxUnpool2d(kernel_size=2, stride=2 )(z, ind1) 
        depth_op = self.depth_dec1(z)

        return task_op, depth_op



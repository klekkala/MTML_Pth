# Required libraries
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

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

        features = list(vgg16_bn.features.children())
        weight = features[0].weight.data 

        # We need the maxpooled indices and the original pretrained model doesn't provide it, hence replacing it.
        features[6], features[13], features[23], features[33], features[43] = maxpool, maxpool, maxpool, maxpool, maxpool
        features[0].weight.data = weight

        # encoder
        self.enc1 = nn.Sequential(*features[0:7])
        self.enc2 = nn.Sequential(*features[7:14])
        self.enc3 = nn.Sequential(*features[14:24])
        self.enc4 = nn.Sequential(*features[24:34])
        self.enc5 = nn.Sequential(*features[34:])


class Depth_Decoder(nn.Module):
    """
    Class for Depth Decoder architecture. The last layer of decoder has just one channel and the activation function is Sigmoid.
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
    Class implementation of Generic task Decoder.
    :params: task = type of task - sn or seg
             output_channels = No. of out_channels
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
    Class implementation of Segmentation Single-task. It inherits Decoder and Encoder base classes.
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
    Class implementation of Segmentation Depth Multi-task. It inherits Segmentation and Depth Decoder base classes.
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


class Task_Depth(Decoder, Encoder):
    """
    Class implementation of Depth Single-task. It inherits Decoder and Encoder base classes.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dec1.dep_dec1_act2 = nn.Sigmoid()

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

class Surface_Normal(Decoder, Encoder):
    """
    Class implementation of Surface Normal Single-task. It inherits Decoder and Encoder base classes.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dec1.sn_dec1_act2 = torch.nn.Tanh()

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


class Surface_Normal_Depth(Depth_Decoder, Surface_Normal):
    """
    Class implementation of Surface Normal Depth Multi-task. It inherits Decoder Depth and Surface Normal base classes.
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
    Class for Vanishing point Model architecture.It inherits Encoder base class.
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
    Class implementation of Vp Depth Multi-task. It inherits Decoder Depth and VP base classes.
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

class SegNet(nn.Module):
    def __init__(self):
        super(SegNet, self).__init__()
        # initialise network parameters
        filter = [64, 128, 256, 512, 512]
        self.class_nb = 14

        # define encoder decoder layers
        self.encoder_block = nn.ModuleList([self.conv_layer([3, filter[0]])])
        self.decoder_block = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        for i in range(4):
            self.encoder_block.append(self.conv_layer([filter[i], filter[i + 1]]))
            self.decoder_block.append(self.conv_layer([filter[i + 1], filter[i]]))

        # define convolution layer
        self.conv_block_enc = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        self.conv_block_dec = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        for i in range(4):
            if i == 0:
                self.conv_block_enc.append(self.conv_layer([filter[i + 1], filter[i + 1]]))
                self.conv_block_dec.append(self.conv_layer([filter[i], filter[i]]))
            else:
                self.conv_block_enc.append(nn.Sequential(self.conv_layer([filter[i + 1], filter[i + 1]]),
                                                         self.conv_layer([filter[i + 1], filter[i + 1]])))
                self.conv_block_dec.append(nn.Sequential(self.conv_layer([filter[i], filter[i]]),
                                                         self.conv_layer([filter[i], filter[i]])))

        # define task attention layers
        self.encoder_att = nn.ModuleList([nn.ModuleList([self.att_layer([filter[0], filter[0], filter[0]])])])
        self.decoder_att = nn.ModuleList([nn.ModuleList([self.att_layer([2 * filter[0], filter[0], filter[0]])])])
        self.encoder_block_att = nn.ModuleList([self.conv_layer([filter[0], filter[1]])])
        self.decoder_block_att = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])

        for j in range(3):
            if j < 2:
                self.encoder_att.append(nn.ModuleList([self.att_layer([filter[0], filter[0], filter[0]])]))
                self.decoder_att.append(nn.ModuleList([self.att_layer([2 * filter[0], filter[0], filter[0]])]))
            for i in range(4):
                self.encoder_att[j].append(self.att_layer([2 * filter[i + 1], filter[i + 1], filter[i + 1]]))
                self.decoder_att[j].append(self.att_layer([filter[i + 1] + filter[i], filter[i], filter[i]]))

        for i in range(4):
            if i < 3:
                self.encoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 2]]))
                self.decoder_block_att.append(self.conv_layer([filter[i + 1], filter[i]]))
            else:
                self.encoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 1]]))
                self.decoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 1]]))

        self.pred_task1 = self.conv_layer([filter[0], self.class_nb], pred=True)
        self.pred_task2 = self.conv_layer([filter[0], 1], pred=True)
        self.pred_task3 = self.conv_layer([filter[0], 3], pred=True)

        # define pooling and unpooling functions
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.up_sampling = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.logsigma = nn.Parameter(torch.FloatTensor([-0.5, -0.5, -0.5]))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def conv_layer(self, channel, pred=False):
        if not pred:
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=channel[1]),
                nn.ReLU(inplace=True),
            )
        else:
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=channel[0], out_channels=channel[0], kernel_size=3, padding=1),
                nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=1, padding=0),
            )
        return conv_block

    def att_layer(self, channel):
        att_block = nn.Sequential(
            nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=1, padding=0),
            nn.BatchNorm2d(channel[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channel[1], out_channels=channel[2], kernel_size=1, padding=0),
            nn.BatchNorm2d(channel[2]),
            nn.Sigmoid(),
        )
        return att_block

    def forward(self, x):
        g_encoder, g_decoder, g_maxpool, g_upsampl, indices = ([0] * 5 for _ in range(5))
        for i in range(5):
            g_encoder[i], g_decoder[-i - 1] = ([0] * 2 for _ in range(2))

        # define attention list for tasks
        atten_encoder, atten_decoder = ([0] * 3 for _ in range(2))
        for i in range(3):
            atten_encoder[i], atten_decoder[i] = ([0] * 5 for _ in range(2))
        for i in range(3):
            for j in range(5):
                atten_encoder[i][j], atten_decoder[i][j] = ([0] * 3 for _ in range(2))

        # define global shared network
        for i in range(5):
            if i == 0:
                g_encoder[i][0] = self.encoder_block[i](x)
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])
            else:
                g_encoder[i][0] = self.encoder_block[i](g_maxpool[i - 1])
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])

        for i in range(5):
            if i == 0:
                g_upsampl[i] = self.up_sampling(g_maxpool[-1], indices[-i - 1])
                g_decoder[i][0] = self.decoder_block[-i - 1](g_upsampl[i])
                g_decoder[i][1] = self.conv_block_dec[-i - 1](g_decoder[i][0])
            else:
                g_upsampl[i] = self.up_sampling(g_decoder[i - 1][-1], indices[-i - 1])
                g_decoder[i][0] = self.decoder_block[-i - 1](g_upsampl[i])
                g_decoder[i][1] = self.conv_block_dec[-i - 1](g_decoder[i][0])

        # define task dependent attention module
        for i in range(3):
            for j in range(5):
                if j == 0:
                    atten_encoder[i][j][0] = self.encoder_att[i][j](g_encoder[j][0])
                    atten_encoder[i][j][1] = (atten_encoder[i][j][0]) * g_encoder[j][1]
                    atten_encoder[i][j][2] = self.encoder_block_att[j](atten_encoder[i][j][1])
                    atten_encoder[i][j][2] = F.max_pool2d(atten_encoder[i][j][2], kernel_size=2, stride=2)
                else:
                    atten_encoder[i][j][0] = self.encoder_att[i][j](torch.cat((g_encoder[j][0], atten_encoder[i][j - 1][2]), dim=1))
                    atten_encoder[i][j][1] = (atten_encoder[i][j][0]) * g_encoder[j][1]
                    atten_encoder[i][j][2] = self.encoder_block_att[j](atten_encoder[i][j][1])
                    atten_encoder[i][j][2] = F.max_pool2d(atten_encoder[i][j][2], kernel_size=2, stride=2)

            for j in range(5):
                if j == 0:
                    atten_decoder[i][j][0] = F.interpolate(atten_encoder[i][-1][-1], scale_factor=2, mode='bilinear', align_corners=True)
                    atten_decoder[i][j][0] = self.decoder_block_att[-j - 1](atten_decoder[i][j][0])
                    atten_decoder[i][j][1] = self.decoder_att[i][-j - 1](torch.cat((g_upsampl[j], atten_decoder[i][j][0]), dim=1))
                    atten_decoder[i][j][2] = (atten_decoder[i][j][1]) * g_decoder[j][-1]
                else:
                    atten_decoder[i][j][0] = F.interpolate(atten_decoder[i][j - 1][2], scale_factor=2, mode='bilinear', align_corners=True)
                    atten_decoder[i][j][0] = self.decoder_block_att[-j - 1](atten_decoder[i][j][0])
                    atten_decoder[i][j][1] = self.decoder_att[i][-j - 1](torch.cat((g_upsampl[j], atten_decoder[i][j][0]), dim=1))
                    atten_decoder[i][j][2] = (atten_decoder[i][j][1]) * g_decoder[j][-1]

        # define task prediction layers
        t1_pred = F.log_softmax(self.pred_task1(atten_decoder[0][-1][-1]), dim=1)
        t2_pred = self.pred_task2(atten_decoder[1][-1][-1])
        # t3_pred = self.pred_task3(atten_decoder[2][-1][-1])
        # t3_pred = t3_pred / torch.norm(t3_pred, p=2, dim=1, keepdim=True)
 
        return [t1_pred, t2_pred], self.logsigma


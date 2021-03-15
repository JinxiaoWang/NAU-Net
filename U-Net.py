from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from inplace_abn import InPlaceABN
except ImportError:
    InPlaceABN = None


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):

        if use_batchnorm == "inplace" and InPlaceABN is None:
            raise RuntimeError(
                "In order to use `use_batchnorm='inplace'` inplace_abn package must be installed. "
                + "To install see: https://github.com/mapillary/inplace_abn"
            )

        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        if use_batchnorm == "inplace":
            bn = InPlaceABN(out_channels, activation="leaky_relu", activation_param=0.0)
            relu = nn.Identity()

        elif use_batchnorm and use_batchnorm != "inplace":
            bn = nn.BatchNorm2d(out_channels)

        else:
            bn = nn.Identity()

        super(Conv2dReLU, self).__init__(conv, bn, relu)

class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            use_batchnorm=True,
            attention_type=None,
    ):
        super().__init__()
        self.conv1 =   Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class Resnet_Layer0_4Band(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False), 
            nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        return self.conv(x) 

class U_Net(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        
        self.base_layers = list(models.resnet50(pretrained=True).children())                
        
        #self.layer0 = nn.Sequential(*self.base_layers[:3]) # if input is 3-band image 
        
        self.layer0 = Resnet_Layer0_4Band(4,64)    #  if input is 4-band image

        self.layer1 = nn.Sequential(*self.base_layers[3:5])     
      
        self.layer2 = self.base_layers[5]         

        self.layer3 = self.base_layers[6]        
 
        self.layer4 = self.base_layers[7]  
        
        self.decode_block3 = DecoderBlock(2048,1024,256)
        self.decode_block2 = DecoderBlock(256,512,128)
        self.decode_block1 = DecoderBlock(128,256,64)
        self.decode_block0 = DecoderBlock(64,64,64)
        self.decode_block_f = DecoderBlock(64,32,32)
        self.shallow = Conv2dReLU(4,32,3,1)

        self.conv_last2 = nn.Conv2d(32,n_class,3,1,1)
        
    def forward(self, input):
        
        layer_shallow = self.shallow(input)
        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        
        x = self.decode_block3(layer4,layer3)
        x = self.decode_block2(x,layer2)
     
        x = self.decode_block1(x,layer1)

        x = self.decode_block0(x,layer0)   

        x = self.decode_block_f(x,layer_shallow)

        out1 = self.conv_last2(x)
        return out1
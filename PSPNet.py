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


class PSPBlock(nn.Module):

    def __init__(self, in_channels, out_channels, pool_size, use_bathcnorm=True):
        super().__init__()
        if pool_size == 1:
            use_bathcnorm = False  # PyTorch does not support BatchNorm for 1x1 shape
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size)),
            Conv2dReLU(in_channels, out_channels, (1, 1), use_batchnorm=use_bathcnorm)
        )

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        x = self.pool(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear',align_corners=True)
        return x


class PSPModule(nn.Module):
    def __init__(self, in_channels, sizes=(1, 2, 3, 6), use_bathcnorm=True):
        super().__init__()

        self.blocks = nn.ModuleList([
            PSPBlock(in_channels, in_channels // len(sizes), size, use_bathcnorm=use_bathcnorm) for size in sizes
        ])

    def forward(self, x):
        xs = [block(x) for block in self.blocks] + [x]
        x = torch.cat(xs, dim=1)
        return x


class PSPDecoder(nn.Module):

    def __init__(
            self,
            encoder_channels,
            use_batchnorm=True,
            out_channels=512,
            dropout=0.2,
    ):
        super().__init__()

        self.psp = PSPModule(
            #in_channels=encoder_channels[-1],
            in_channels = encoder_channels,
            sizes=(1, 2, 3, 6),
            use_bathcnorm=use_batchnorm,
        )

        self.conv = Conv2dReLU(
            #in_channels=encoder_channels[-1] * 2,
            in_channels=encoder_channels*2,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, *features):
        x = features[-1]
        x = self.psp(x)
        x = self.conv(x)
        #x = self.dropout(x)

        return x

    
from pytorchcv.model_provider import get_model as ptcv_get_model

net = ptcv_get_model("resnetd50b", pretrained=True)
BaseLayer = list(net.children())[0]
SEInitBlock = list(BaseLayer[0].children())

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

class PSPNet(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        
        self.layer0 = Resnet_Layer0_4Band(4,128) 

        self.layer1 = nn.Sequential(SEInitBlock[3],BaseLayer[1])       
      
        self.layer2 = BaseLayer[2]          

        self.layer3 = BaseLayer[3]  
        
        self.layer4 = BaseLayer[4]
        
        self.psp = PSPDecoder(2048,512)
 
        self.conv_last2 = Conv2dReLU(512,64,3,1)
        self.conv_last3 = nn.Conv2d(64,n_class,3,1,1)
        
        
        

    def forward(self, input):
        
        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        x = self.psp(layer4)
        x = self.conv_last2(x)
        out = self.conv_last3(x)      
        
        out = F.interpolate(out, scale_factor=8, mode="bilinear",align_corners=True)

        return out
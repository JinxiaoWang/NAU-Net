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


class NAU_Net(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        
        self.base_layers = list(models.resnet50(pretrained=True).children())                
        
        self.layer0 = nn.Sequential(*self.base_layers[:3]) 

        self.layer1 = nn.Sequential(*self.base_layers[3:5])       
      
        self.layer2 = self.base_layers[5]         

        self.layer3 = self.base_layers[6]         
 
        self.layer4 = self.base_layers[7] 

        self.decode_block3 = DecoderBlock(2048,1024,256)
        self.decode_block2 = DecoderBlock(256,512,128)
        self.decode_block1 = DecoderBlock(128,256,64)

        self.decode_block0 = DecoderBlock(64,64,64)
        self.decode_block_f = DecoderBlock(64,32,32)

        self.shallow = Conv2dReLU(3,32,3,1)

        self.conv_last2 = nn.Conv2d(32,n_class,3,1,1)
        
        self.alpha0 = nn.Parameter(torch.ones(1))
        self.alpha1 = nn.Parameter(torch.ones(1))
        self.alpha2 = nn.Parameter(torch.ones(1))
        self.alpha3 = nn.Parameter(torch.ones(1))
        self.beita0 = nn.Parameter(torch.zeros(1))
        self.beita1 = nn.Parameter(torch.zeros(1))
        self.beita2 = nn.Parameter(torch.zeros(1))
        self.beita3 = nn.Parameter(torch.zeros(1))
        
    def forward(self, input):
        
        layer_shallow = self.shallow(input[:,:-1,:,:])
        layer0 = self.layer0(input[:,:-1,:,:])
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        
        ndi = input[:,-1:,:,:]

        #ndi3 = F.avg_pool2d(ndi,kernel_size = [16,16])
        #layer3 = self.alpha0*ndi3*layer3 + self.beita0*layer3  # No NA block is applied to layer3 in the updated NAU-Net architecture.
        x = self.decode_block3(layer4,layer3)
        
        ndi2 = F.avg_pool2d(ndi,kernel_size = [8,8])
        layer2 = self.alpha1*ndi2*layer2 + self.beita1*layer2
        x = self.decode_block2(x,layer2)

        ndi1 = F.avg_pool2d(ndi,kernel_size = [4,4])
        layer1 = self.alpha2*ndi1*layer1 + self.beita2*layer1      
        x = self.decode_block1(x,layer1)
 
        ndi0 = F.avg_pool2d(ndi,kernel_size = [2,2])
        layer0 = self.alpha3*ndi0*layer0 + self.beita3*layer0
        x = self.decode_block0(x,layer0) 

        x = self.decode_block_f(x,layer_shallow)

        out1 = self.conv_last2(x)
        return out1

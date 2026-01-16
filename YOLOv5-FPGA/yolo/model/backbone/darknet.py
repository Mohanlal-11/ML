from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn

from .utils import Conv, ConcatBlock

class SplitSpatial(nn.Module):
    def __init__(self,in_ch):
        super(SplitSpatial, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch, kernel_size=2, stride=2, bias=False).requires_grad_(False)  
        self.conv2 = nn.Conv2d(in_ch, in_ch, kernel_size=2, stride=2, bias=False).requires_grad_(False)  
        self.conv3 = nn.Conv2d(in_ch, in_ch, kernel_size=2, stride=2, bias=False).requires_grad_(False)  
        self.conv4 = nn.Conv2d(in_ch, in_ch, kernel_size=2, stride=2, bias=False).requires_grad_(False)

        with torch.no_grad():
            wts1 = torch.zeros(in_ch, in_ch, 2,2)
            wts2 = torch.zeros(in_ch, in_ch, 2,2)
            wts3 = torch.zeros(in_ch, in_ch, 2,2)
            wts4 = torch.zeros(in_ch, in_ch, 2,2)
            for i in range(in_ch):
                wts1[i, i, 0, 0] = 1
                wts2[i, i, 1, 0] = 1
                wts3[i, i, 0, 1] = 1
                wts4[i, i, 1, 1] = 1

            self.conv1.weight.copy_(wts1)
            self.conv2.weight.copy_(wts2)
            self.conv3.weight.copy_(wts3)
            self.conv4.weight.copy_(wts4)
            
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        return torch.cat((x1, x2, x3, x4), dim=1)
    

class SpatialPyramidPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = in_channels // 2
        self.conv1 = Conv(in_channels, mid_channels, 1)
        self.conv2 = Conv(4 * mid_channels, out_channels, 1)
        
        self.pool1 = nn.MaxPool2d(5, 1, 2)
        self.pool2 = nn.MaxPool2d(9, 1, 4)
        self.pool3 = nn.MaxPool2d(13, 1, 6)
        
    def forward(self, x):
        x = self.conv1(x)
        #x1 = F.max_pool2d(x, 5, 1, 2)
        #x2 = F.max_pool2d(x, 9, 1, 4)
        #x3 = F.max_pool2d(x, 13, 1, 6)
        
        x1 = self.pool1(x)
        x2 = self.pool2(x)
        x3 = self.pool3(x)
        
        out = torch.cat((x, x1, x2, x3), dim=1)
        out = self.conv2(out)
        return out
    
    
class Focus(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = Conv(4 * in_channels, out_channels, kernel_size)
        self.out_channels = out_channels
        self.split_spatial = SplitSpatial(in_ch=in_channels)

    def forward(self, x):
        # concat = torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1)
        concat = self.split_spatial(x)
        return self.conv(concat)
    
    
class CSPDarknet(nn.Sequential):
    def __init__(self, out_channels_list, layers):
        assert len(layers) + 2 == len(out_channels_list), "len(layers) != len(out_channels_list)"

        d = OrderedDict()
        d["layer0"] = Focus(3, out_channels_list[0], 3)
        
        for i, ch in enumerate(out_channels_list[1:]):
            in_channels = out_channels_list[i]
            name = "layer{}".format(i + 1)
            d[name] = nn.Sequential(Conv(in_channels, ch, 3, 2))
            if i < len(out_channels_list) - 2:
                d[name].add_module("concat", ConcatBlock(ch, ch, layers[i], True))
            else:
                d[name].add_module("spp", SpatialPyramidPooling(ch, ch))
            
        super().__init__(d)
        
        #for m in self.modules():
        #    if isinstance(m, nn.Conv2d):
        #        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
        #    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #        nn.init.constant_(m.weight, 1)
        #        nn.init.constant_(m.bias, 0)
           
        
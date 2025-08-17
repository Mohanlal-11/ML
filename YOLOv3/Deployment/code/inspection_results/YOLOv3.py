# GENETARED BY NNDCT, DO NOT EDIT!

import torch
from torch import tensor
import pytorch_nndct as py_nndct

class YOLOv3(py_nndct.nn.NndctQuantModel):
    def __init__(self):
        super(YOLOv3, self).__init__()
        self.module_0 = py_nndct.nn.Input() #YOLOv3::input_0
        self.module_1 = py_nndct.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Darknet53[backbone]/ConvLayer[conv1]/Sequential[conv]/Conv2d[0]/input.3
        self.module_2 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Darknet53[backbone]/ConvLayer[conv1]/Sequential[conv]/LeakyReLU[2]/input.7
        self.module_3 = py_nndct.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv2]/ConvLayer[0]/Sequential[conv]/Conv2d[0]/input.9
        self.module_4 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv2]/ConvLayer[0]/Sequential[conv]/LeakyReLU[2]/input.13
        self.module_5 = py_nndct.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv2]/Residual_Block[1]/Sequential[skip_connection]/ConvLayer[0]/Sequential[conv]/Conv2d[0]/input.15
        self.module_6 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv2]/Residual_Block[1]/Sequential[skip_connection]/ConvLayer[0]/Sequential[conv]/LeakyReLU[2]/input.19
        self.module_7 = py_nndct.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv2]/Residual_Block[1]/Sequential[skip_connection]/ConvLayer[1]/Sequential[conv]/Conv2d[0]/input.21
        self.module_8 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv2]/Residual_Block[1]/Sequential[skip_connection]/ConvLayer[1]/Sequential[conv]/LeakyReLU[2]/15862
        self.module_9 = py_nndct.nn.Add() #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv2]/Residual_Block[1]/input.25
        self.module_10 = py_nndct.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv3]/ConvLayer[0]/Sequential[conv]/Conv2d[0]/input.27
        self.module_11 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv3]/ConvLayer[0]/Sequential[conv]/LeakyReLU[2]/input.31
        self.module_12 = py_nndct.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv3]/Residual_Block[1]/Sequential[skip_connection]/ConvLayer[0]/Sequential[conv]/Conv2d[0]/input.33
        self.module_13 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv3]/Residual_Block[1]/Sequential[skip_connection]/ConvLayer[0]/Sequential[conv]/LeakyReLU[2]/input.37
        self.module_14 = py_nndct.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv3]/Residual_Block[1]/Sequential[skip_connection]/ConvLayer[1]/Sequential[conv]/Conv2d[0]/input.39
        self.module_15 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv3]/Residual_Block[1]/Sequential[skip_connection]/ConvLayer[1]/Sequential[conv]/LeakyReLU[2]/15945
        self.module_16 = py_nndct.nn.Add() #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv3]/Residual_Block[1]/input.43
        self.module_17 = py_nndct.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv3]/Residual_Block[2]/Sequential[skip_connection]/ConvLayer[0]/Sequential[conv]/Conv2d[0]/input.45
        self.module_18 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv3]/Residual_Block[2]/Sequential[skip_connection]/ConvLayer[0]/Sequential[conv]/LeakyReLU[2]/input.49
        self.module_19 = py_nndct.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv3]/Residual_Block[2]/Sequential[skip_connection]/ConvLayer[1]/Sequential[conv]/Conv2d[0]/input.51
        self.module_20 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv3]/Residual_Block[2]/Sequential[skip_connection]/ConvLayer[1]/Sequential[conv]/LeakyReLU[2]/16001
        self.module_21 = py_nndct.nn.Add() #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv3]/Residual_Block[2]/input.55
        self.module_22 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv4]/ConvLayer[0]/Sequential[conv]/Conv2d[0]/input.57
        self.module_23 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv4]/ConvLayer[0]/Sequential[conv]/LeakyReLU[2]/input.61
        self.module_24 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv4]/Residual_Block[1]/Sequential[skip_connection]/ConvLayer[0]/Sequential[conv]/Conv2d[0]/input.63
        self.module_25 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv4]/Residual_Block[1]/Sequential[skip_connection]/ConvLayer[0]/Sequential[conv]/LeakyReLU[2]/input.67
        self.module_26 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv4]/Residual_Block[1]/Sequential[skip_connection]/ConvLayer[1]/Sequential[conv]/Conv2d[0]/input.69
        self.module_27 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv4]/Residual_Block[1]/Sequential[skip_connection]/ConvLayer[1]/Sequential[conv]/LeakyReLU[2]/16084
        self.module_28 = py_nndct.nn.Add() #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv4]/Residual_Block[1]/input.73
        self.module_29 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv4]/Residual_Block[2]/Sequential[skip_connection]/ConvLayer[0]/Sequential[conv]/Conv2d[0]/input.75
        self.module_30 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv4]/Residual_Block[2]/Sequential[skip_connection]/ConvLayer[0]/Sequential[conv]/LeakyReLU[2]/input.79
        self.module_31 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv4]/Residual_Block[2]/Sequential[skip_connection]/ConvLayer[1]/Sequential[conv]/Conv2d[0]/input.81
        self.module_32 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv4]/Residual_Block[2]/Sequential[skip_connection]/ConvLayer[1]/Sequential[conv]/LeakyReLU[2]/16140
        self.module_33 = py_nndct.nn.Add() #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv4]/Residual_Block[2]/input.85
        self.module_34 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv4]/Residual_Block[3]/Sequential[skip_connection]/ConvLayer[0]/Sequential[conv]/Conv2d[0]/input.87
        self.module_35 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv4]/Residual_Block[3]/Sequential[skip_connection]/ConvLayer[0]/Sequential[conv]/LeakyReLU[2]/input.91
        self.module_36 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv4]/Residual_Block[3]/Sequential[skip_connection]/ConvLayer[1]/Sequential[conv]/Conv2d[0]/input.93
        self.module_37 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv4]/Residual_Block[3]/Sequential[skip_connection]/ConvLayer[1]/Sequential[conv]/LeakyReLU[2]/16196
        self.module_38 = py_nndct.nn.Add() #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv4]/Residual_Block[3]/input.97
        self.module_39 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv4]/Residual_Block[4]/Sequential[skip_connection]/ConvLayer[0]/Sequential[conv]/Conv2d[0]/input.99
        self.module_40 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv4]/Residual_Block[4]/Sequential[skip_connection]/ConvLayer[0]/Sequential[conv]/LeakyReLU[2]/input.103
        self.module_41 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv4]/Residual_Block[4]/Sequential[skip_connection]/ConvLayer[1]/Sequential[conv]/Conv2d[0]/input.105
        self.module_42 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv4]/Residual_Block[4]/Sequential[skip_connection]/ConvLayer[1]/Sequential[conv]/LeakyReLU[2]/16252
        self.module_43 = py_nndct.nn.Add() #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv4]/Residual_Block[4]/input.109
        self.module_44 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv4]/Residual_Block[5]/Sequential[skip_connection]/ConvLayer[0]/Sequential[conv]/Conv2d[0]/input.111
        self.module_45 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv4]/Residual_Block[5]/Sequential[skip_connection]/ConvLayer[0]/Sequential[conv]/LeakyReLU[2]/input.115
        self.module_46 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv4]/Residual_Block[5]/Sequential[skip_connection]/ConvLayer[1]/Sequential[conv]/Conv2d[0]/input.117
        self.module_47 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv4]/Residual_Block[5]/Sequential[skip_connection]/ConvLayer[1]/Sequential[conv]/LeakyReLU[2]/16308
        self.module_48 = py_nndct.nn.Add() #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv4]/Residual_Block[5]/input.121
        self.module_49 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv4]/Residual_Block[6]/Sequential[skip_connection]/ConvLayer[0]/Sequential[conv]/Conv2d[0]/input.123
        self.module_50 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv4]/Residual_Block[6]/Sequential[skip_connection]/ConvLayer[0]/Sequential[conv]/LeakyReLU[2]/input.127
        self.module_51 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv4]/Residual_Block[6]/Sequential[skip_connection]/ConvLayer[1]/Sequential[conv]/Conv2d[0]/input.129
        self.module_52 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv4]/Residual_Block[6]/Sequential[skip_connection]/ConvLayer[1]/Sequential[conv]/LeakyReLU[2]/16364
        self.module_53 = py_nndct.nn.Add() #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv4]/Residual_Block[6]/input.133
        self.module_54 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv4]/Residual_Block[7]/Sequential[skip_connection]/ConvLayer[0]/Sequential[conv]/Conv2d[0]/input.135
        self.module_55 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv4]/Residual_Block[7]/Sequential[skip_connection]/ConvLayer[0]/Sequential[conv]/LeakyReLU[2]/input.139
        self.module_56 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv4]/Residual_Block[7]/Sequential[skip_connection]/ConvLayer[1]/Sequential[conv]/Conv2d[0]/input.141
        self.module_57 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv4]/Residual_Block[7]/Sequential[skip_connection]/ConvLayer[1]/Sequential[conv]/LeakyReLU[2]/16420
        self.module_58 = py_nndct.nn.Add() #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv4]/Residual_Block[7]/input.145
        self.module_59 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv4]/Residual_Block[8]/Sequential[skip_connection]/ConvLayer[0]/Sequential[conv]/Conv2d[0]/input.147
        self.module_60 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv4]/Residual_Block[8]/Sequential[skip_connection]/ConvLayer[0]/Sequential[conv]/LeakyReLU[2]/input.151
        self.module_61 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv4]/Residual_Block[8]/Sequential[skip_connection]/ConvLayer[1]/Sequential[conv]/Conv2d[0]/input.153
        self.module_62 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv4]/Residual_Block[8]/Sequential[skip_connection]/ConvLayer[1]/Sequential[conv]/LeakyReLU[2]/16476
        self.module_63 = py_nndct.nn.Add() #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv4]/Residual_Block[8]/input.157
        self.module_64 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv5]/ConvLayer[0]/Sequential[conv]/Conv2d[0]/input.159
        self.module_65 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv5]/ConvLayer[0]/Sequential[conv]/LeakyReLU[2]/input.163
        self.module_66 = py_nndct.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv5]/Residual_Block[1]/Sequential[skip_connection]/ConvLayer[0]/Sequential[conv]/Conv2d[0]/input.165
        self.module_67 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv5]/Residual_Block[1]/Sequential[skip_connection]/ConvLayer[0]/Sequential[conv]/LeakyReLU[2]/input.169
        self.module_68 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv5]/Residual_Block[1]/Sequential[skip_connection]/ConvLayer[1]/Sequential[conv]/Conv2d[0]/input.171
        self.module_69 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv5]/Residual_Block[1]/Sequential[skip_connection]/ConvLayer[1]/Sequential[conv]/LeakyReLU[2]/16559
        self.module_70 = py_nndct.nn.Add() #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv5]/Residual_Block[1]/input.175
        self.module_71 = py_nndct.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv5]/Residual_Block[2]/Sequential[skip_connection]/ConvLayer[0]/Sequential[conv]/Conv2d[0]/input.177
        self.module_72 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv5]/Residual_Block[2]/Sequential[skip_connection]/ConvLayer[0]/Sequential[conv]/LeakyReLU[2]/input.181
        self.module_73 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv5]/Residual_Block[2]/Sequential[skip_connection]/ConvLayer[1]/Sequential[conv]/Conv2d[0]/input.183
        self.module_74 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv5]/Residual_Block[2]/Sequential[skip_connection]/ConvLayer[1]/Sequential[conv]/LeakyReLU[2]/16615
        self.module_75 = py_nndct.nn.Add() #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv5]/Residual_Block[2]/input.187
        self.module_76 = py_nndct.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv5]/Residual_Block[3]/Sequential[skip_connection]/ConvLayer[0]/Sequential[conv]/Conv2d[0]/input.189
        self.module_77 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv5]/Residual_Block[3]/Sequential[skip_connection]/ConvLayer[0]/Sequential[conv]/LeakyReLU[2]/input.193
        self.module_78 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv5]/Residual_Block[3]/Sequential[skip_connection]/ConvLayer[1]/Sequential[conv]/Conv2d[0]/input.195
        self.module_79 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv5]/Residual_Block[3]/Sequential[skip_connection]/ConvLayer[1]/Sequential[conv]/LeakyReLU[2]/16671
        self.module_80 = py_nndct.nn.Add() #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv5]/Residual_Block[3]/input.199
        self.module_81 = py_nndct.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv5]/Residual_Block[4]/Sequential[skip_connection]/ConvLayer[0]/Sequential[conv]/Conv2d[0]/input.201
        self.module_82 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv5]/Residual_Block[4]/Sequential[skip_connection]/ConvLayer[0]/Sequential[conv]/LeakyReLU[2]/input.205
        self.module_83 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv5]/Residual_Block[4]/Sequential[skip_connection]/ConvLayer[1]/Sequential[conv]/Conv2d[0]/input.207
        self.module_84 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv5]/Residual_Block[4]/Sequential[skip_connection]/ConvLayer[1]/Sequential[conv]/LeakyReLU[2]/16727
        self.module_85 = py_nndct.nn.Add() #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv5]/Residual_Block[4]/input.211
        self.module_86 = py_nndct.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv5]/Residual_Block[5]/Sequential[skip_connection]/ConvLayer[0]/Sequential[conv]/Conv2d[0]/input.213
        self.module_87 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv5]/Residual_Block[5]/Sequential[skip_connection]/ConvLayer[0]/Sequential[conv]/LeakyReLU[2]/input.217
        self.module_88 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv5]/Residual_Block[5]/Sequential[skip_connection]/ConvLayer[1]/Sequential[conv]/Conv2d[0]/input.219
        self.module_89 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv5]/Residual_Block[5]/Sequential[skip_connection]/ConvLayer[1]/Sequential[conv]/LeakyReLU[2]/16783
        self.module_90 = py_nndct.nn.Add() #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv5]/Residual_Block[5]/input.223
        self.module_91 = py_nndct.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv5]/Residual_Block[6]/Sequential[skip_connection]/ConvLayer[0]/Sequential[conv]/Conv2d[0]/input.225
        self.module_92 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv5]/Residual_Block[6]/Sequential[skip_connection]/ConvLayer[0]/Sequential[conv]/LeakyReLU[2]/input.229
        self.module_93 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv5]/Residual_Block[6]/Sequential[skip_connection]/ConvLayer[1]/Sequential[conv]/Conv2d[0]/input.231
        self.module_94 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv5]/Residual_Block[6]/Sequential[skip_connection]/ConvLayer[1]/Sequential[conv]/LeakyReLU[2]/16839
        self.module_95 = py_nndct.nn.Add() #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv5]/Residual_Block[6]/input.235
        self.module_96 = py_nndct.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv5]/Residual_Block[7]/Sequential[skip_connection]/ConvLayer[0]/Sequential[conv]/Conv2d[0]/input.237
        self.module_97 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv5]/Residual_Block[7]/Sequential[skip_connection]/ConvLayer[0]/Sequential[conv]/LeakyReLU[2]/input.241
        self.module_98 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv5]/Residual_Block[7]/Sequential[skip_connection]/ConvLayer[1]/Sequential[conv]/Conv2d[0]/input.243
        self.module_99 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv5]/Residual_Block[7]/Sequential[skip_connection]/ConvLayer[1]/Sequential[conv]/LeakyReLU[2]/16895
        self.module_100 = py_nndct.nn.Add() #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv5]/Residual_Block[7]/input.247
        self.module_101 = py_nndct.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv5]/Residual_Block[8]/Sequential[skip_connection]/ConvLayer[0]/Sequential[conv]/Conv2d[0]/input.249
        self.module_102 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv5]/Residual_Block[8]/Sequential[skip_connection]/ConvLayer[0]/Sequential[conv]/LeakyReLU[2]/input.253
        self.module_103 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv5]/Residual_Block[8]/Sequential[skip_connection]/ConvLayer[1]/Sequential[conv]/Conv2d[0]/input.255
        self.module_104 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv5]/Residual_Block[8]/Sequential[skip_connection]/ConvLayer[1]/Sequential[conv]/LeakyReLU[2]/16951
        self.module_105 = py_nndct.nn.Add() #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv5]/Residual_Block[8]/input.259
        self.module_106 = py_nndct.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv6]/ConvLayer[0]/Sequential[conv]/Conv2d[0]/input.261
        self.module_107 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv6]/ConvLayer[0]/Sequential[conv]/LeakyReLU[2]/input.265
        self.module_108 = py_nndct.nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv6]/Residual_Block[1]/Sequential[skip_connection]/ConvLayer[0]/Sequential[conv]/Conv2d[0]/input.267
        self.module_109 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv6]/Residual_Block[1]/Sequential[skip_connection]/ConvLayer[0]/Sequential[conv]/LeakyReLU[2]/input.271
        self.module_110 = py_nndct.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv6]/Residual_Block[1]/Sequential[skip_connection]/ConvLayer[1]/Sequential[conv]/Conv2d[0]/input.273
        self.module_111 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv6]/Residual_Block[1]/Sequential[skip_connection]/ConvLayer[1]/Sequential[conv]/LeakyReLU[2]/17034
        self.module_112 = py_nndct.nn.Add() #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv6]/Residual_Block[1]/input.277
        self.module_113 = py_nndct.nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv6]/Residual_Block[2]/Sequential[skip_connection]/ConvLayer[0]/Sequential[conv]/Conv2d[0]/input.279
        self.module_114 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv6]/Residual_Block[2]/Sequential[skip_connection]/ConvLayer[0]/Sequential[conv]/LeakyReLU[2]/input.283
        self.module_115 = py_nndct.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv6]/Residual_Block[2]/Sequential[skip_connection]/ConvLayer[1]/Sequential[conv]/Conv2d[0]/input.285
        self.module_116 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv6]/Residual_Block[2]/Sequential[skip_connection]/ConvLayer[1]/Sequential[conv]/LeakyReLU[2]/17090
        self.module_117 = py_nndct.nn.Add() #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv6]/Residual_Block[2]/input.289
        self.module_118 = py_nndct.nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv6]/Residual_Block[3]/Sequential[skip_connection]/ConvLayer[0]/Sequential[conv]/Conv2d[0]/input.291
        self.module_119 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv6]/Residual_Block[3]/Sequential[skip_connection]/ConvLayer[0]/Sequential[conv]/LeakyReLU[2]/input.295
        self.module_120 = py_nndct.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv6]/Residual_Block[3]/Sequential[skip_connection]/ConvLayer[1]/Sequential[conv]/Conv2d[0]/input.297
        self.module_121 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv6]/Residual_Block[3]/Sequential[skip_connection]/ConvLayer[1]/Sequential[conv]/LeakyReLU[2]/17146
        self.module_122 = py_nndct.nn.Add() #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv6]/Residual_Block[3]/input.301
        self.module_123 = py_nndct.nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv6]/Residual_Block[4]/Sequential[skip_connection]/ConvLayer[0]/Sequential[conv]/Conv2d[0]/input.303
        self.module_124 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv6]/Residual_Block[4]/Sequential[skip_connection]/ConvLayer[0]/Sequential[conv]/LeakyReLU[2]/input.307
        self.module_125 = py_nndct.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv6]/Residual_Block[4]/Sequential[skip_connection]/ConvLayer[1]/Sequential[conv]/Conv2d[0]/input.309
        self.module_126 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv6]/Residual_Block[4]/Sequential[skip_connection]/ConvLayer[1]/Sequential[conv]/LeakyReLU[2]/17202
        self.module_127 = py_nndct.nn.Add() #YOLOv3::YOLOv3/Darknet53[backbone]/Sequential[conv6]/Residual_Block[4]/input.313
        self.module_128 = py_nndct.nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Sequential[layer1]/ConvLayer[0]/Sequential[conv]/Conv2d[0]/input.315
        self.module_129 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Sequential[layer1]/ConvLayer[0]/Sequential[conv]/LeakyReLU[2]/input.319
        self.module_130 = py_nndct.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Sequential[layer1]/ConvLayer[1]/Sequential[conv]/Conv2d[0]/input.321
        self.module_131 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Sequential[layer1]/ConvLayer[1]/Sequential[conv]/LeakyReLU[2]/input.325
        self.module_132 = py_nndct.nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Sequential[layer1]/ConvLayer[2]/Sequential[conv]/Conv2d[0]/input.327
        self.module_133 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Sequential[layer1]/ConvLayer[2]/Sequential[conv]/LeakyReLU[2]/input.331
        self.module_134 = py_nndct.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Sequential[layer1]/ConvLayer[3]/Sequential[conv]/Conv2d[0]/input.333
        self.module_135 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Sequential[layer1]/ConvLayer[3]/Sequential[conv]/LeakyReLU[2]/input.337
        self.module_136 = py_nndct.nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Sequential[layer1]/ConvLayer[4]/Sequential[conv]/Conv2d[0]/input.339
        self.module_137 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Sequential[layer1]/ConvLayer[4]/Sequential[conv]/LeakyReLU[2]/input.343
        self.module_138 = py_nndct.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Sequential[stage1]/ConvLayer[0]/Sequential[conv]/Conv2d[0]/input.345
        self.module_139 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Sequential[stage1]/ConvLayer[0]/Sequential[conv]/LeakyReLU[2]/input.349
        self.module_140 = py_nndct.nn.Conv2d(in_channels=1024, out_channels=255, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Sequential[stage1]/Conv2d[1]/17389
        self.module_141 = py_nndct.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Sequential[upsample1]/ConvLayer[0]/Sequential[conv]/Conv2d[0]/input.351
        self.module_142 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Sequential[upsample1]/ConvLayer[0]/Sequential[conv]/LeakyReLU[2]/input.355
        self.module_143 = py_nndct.nn.Interpolate() #YOLOv3::YOLOv3/Sequential[upsample1]/UpsamplingNearest2d[1]/17421
        self.module_144 = py_nndct.nn.Cat() #YOLOv3::YOLOv3/input.357
        self.module_145 = py_nndct.nn.Conv2d(in_channels=768, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Sequential[layer2]/ConvLayer[0]/Sequential[conv]/Conv2d[0]/input.359
        self.module_146 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Sequential[layer2]/ConvLayer[0]/Sequential[conv]/LeakyReLU[2]/input.363
        self.module_147 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Sequential[layer2]/ConvLayer[1]/Sequential[conv]/Conv2d[0]/input.365
        self.module_148 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Sequential[layer2]/ConvLayer[1]/Sequential[conv]/LeakyReLU[2]/input.369
        self.module_149 = py_nndct.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Sequential[layer2]/ConvLayer[2]/Sequential[conv]/Conv2d[0]/input.371
        self.module_150 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Sequential[layer2]/ConvLayer[2]/Sequential[conv]/LeakyReLU[2]/input.375
        self.module_151 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Sequential[layer2]/ConvLayer[3]/Sequential[conv]/Conv2d[0]/input.377
        self.module_152 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Sequential[layer2]/ConvLayer[3]/Sequential[conv]/LeakyReLU[2]/input.381
        self.module_153 = py_nndct.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Sequential[layer2]/ConvLayer[4]/Sequential[conv]/Conv2d[0]/input.383
        self.module_154 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Sequential[layer2]/ConvLayer[4]/Sequential[conv]/LeakyReLU[2]/input.387
        self.module_155 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Sequential[stage2]/ConvLayer[0]/Sequential[conv]/Conv2d[0]/input.389
        self.module_156 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Sequential[stage2]/ConvLayer[0]/Sequential[conv]/LeakyReLU[2]/input.393
        self.module_157 = py_nndct.nn.Conv2d(in_channels=512, out_channels=255, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Sequential[stage2]/Conv2d[1]/17605
        self.module_158 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Sequential[upsample2]/ConvLayer[0]/Sequential[conv]/Conv2d[0]/input.395
        self.module_159 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Sequential[upsample2]/ConvLayer[0]/Sequential[conv]/LeakyReLU[2]/input.399
        self.module_160 = py_nndct.nn.Interpolate() #YOLOv3::YOLOv3/Sequential[upsample2]/UpsamplingNearest2d[1]/17637
        self.module_161 = py_nndct.nn.Cat() #YOLOv3::YOLOv3/input.401
        self.module_162 = py_nndct.nn.Conv2d(in_channels=384, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Sequential[layer3]/ConvLayer[0]/Sequential[conv]/Conv2d[0]/input.403
        self.module_163 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Sequential[layer3]/ConvLayer[0]/Sequential[conv]/LeakyReLU[2]/input.407
        self.module_164 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Sequential[layer3]/ConvLayer[1]/Sequential[conv]/Conv2d[0]/input.409
        self.module_165 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Sequential[layer3]/ConvLayer[1]/Sequential[conv]/LeakyReLU[2]/input.413
        self.module_166 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Sequential[layer3]/ConvLayer[2]/Sequential[conv]/Conv2d[0]/input.415
        self.module_167 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Sequential[layer3]/ConvLayer[2]/Sequential[conv]/LeakyReLU[2]/input.419
        self.module_168 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Sequential[layer3]/ConvLayer[3]/Sequential[conv]/Conv2d[0]/input.421
        self.module_169 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Sequential[layer3]/ConvLayer[3]/Sequential[conv]/LeakyReLU[2]/input.425
        self.module_170 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Sequential[layer3]/ConvLayer[4]/Sequential[conv]/Conv2d[0]/input.427
        self.module_171 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Sequential[layer3]/ConvLayer[4]/Sequential[conv]/LeakyReLU[2]/input.431
        self.module_172 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Sequential[stage3]/ConvLayer[0]/Sequential[conv]/Conv2d[0]/input.433
        self.module_173 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #YOLOv3::YOLOv3/Sequential[stage3]/ConvLayer[0]/Sequential[conv]/LeakyReLU[2]/input
        self.module_174 = py_nndct.nn.Conv2d(in_channels=256, out_channels=255, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOv3::YOLOv3/Sequential[stage3]/Conv2d[1]/17821

    @py_nndct.nn.forward_processor
    def forward(self, *args):
        output_module_0 = self.module_0(input=args[0])
        output_module_0 = self.module_1(output_module_0)
        output_module_0 = self.module_2(output_module_0)
        output_module_0 = self.module_3(output_module_0)
        output_module_0 = self.module_4(output_module_0)
        output_module_5 = self.module_5(output_module_0)
        output_module_5 = self.module_6(output_module_5)
        output_module_5 = self.module_7(output_module_5)
        output_module_5 = self.module_8(output_module_5)
        output_module_5 = self.module_9(input=output_module_5, other=output_module_0, alpha=1)
        output_module_5 = self.module_10(output_module_5)
        output_module_5 = self.module_11(output_module_5)
        output_module_12 = self.module_12(output_module_5)
        output_module_12 = self.module_13(output_module_12)
        output_module_12 = self.module_14(output_module_12)
        output_module_12 = self.module_15(output_module_12)
        output_module_12 = self.module_16(input=output_module_12, other=output_module_5, alpha=1)
        output_module_17 = self.module_17(output_module_12)
        output_module_17 = self.module_18(output_module_17)
        output_module_17 = self.module_19(output_module_17)
        output_module_17 = self.module_20(output_module_17)
        output_module_17 = self.module_21(input=output_module_17, other=output_module_12, alpha=1)
        output_module_17 = self.module_22(output_module_17)
        output_module_17 = self.module_23(output_module_17)
        output_module_24 = self.module_24(output_module_17)
        output_module_24 = self.module_25(output_module_24)
        output_module_24 = self.module_26(output_module_24)
        output_module_24 = self.module_27(output_module_24)
        output_module_24 = self.module_28(input=output_module_24, other=output_module_17, alpha=1)
        output_module_29 = self.module_29(output_module_24)
        output_module_29 = self.module_30(output_module_29)
        output_module_29 = self.module_31(output_module_29)
        output_module_29 = self.module_32(output_module_29)
        output_module_29 = self.module_33(input=output_module_29, other=output_module_24, alpha=1)
        output_module_34 = self.module_34(output_module_29)
        output_module_34 = self.module_35(output_module_34)
        output_module_34 = self.module_36(output_module_34)
        output_module_34 = self.module_37(output_module_34)
        output_module_34 = self.module_38(input=output_module_34, other=output_module_29, alpha=1)
        output_module_39 = self.module_39(output_module_34)
        output_module_39 = self.module_40(output_module_39)
        output_module_39 = self.module_41(output_module_39)
        output_module_39 = self.module_42(output_module_39)
        output_module_39 = self.module_43(input=output_module_39, other=output_module_34, alpha=1)
        output_module_44 = self.module_44(output_module_39)
        output_module_44 = self.module_45(output_module_44)
        output_module_44 = self.module_46(output_module_44)
        output_module_44 = self.module_47(output_module_44)
        output_module_44 = self.module_48(input=output_module_44, other=output_module_39, alpha=1)
        output_module_49 = self.module_49(output_module_44)
        output_module_49 = self.module_50(output_module_49)
        output_module_49 = self.module_51(output_module_49)
        output_module_49 = self.module_52(output_module_49)
        output_module_49 = self.module_53(input=output_module_49, other=output_module_44, alpha=1)
        output_module_54 = self.module_54(output_module_49)
        output_module_54 = self.module_55(output_module_54)
        output_module_54 = self.module_56(output_module_54)
        output_module_54 = self.module_57(output_module_54)
        output_module_54 = self.module_58(input=output_module_54, other=output_module_49, alpha=1)
        output_module_59 = self.module_59(output_module_54)
        output_module_59 = self.module_60(output_module_59)
        output_module_59 = self.module_61(output_module_59)
        output_module_59 = self.module_62(output_module_59)
        output_module_59 = self.module_63(input=output_module_59, other=output_module_54, alpha=1)
        output_module_64 = self.module_64(output_module_59)
        output_module_64 = self.module_65(output_module_64)
        output_module_66 = self.module_66(output_module_64)
        output_module_66 = self.module_67(output_module_66)
        output_module_66 = self.module_68(output_module_66)
        output_module_66 = self.module_69(output_module_66)
        output_module_66 = self.module_70(input=output_module_66, other=output_module_64, alpha=1)
        output_module_71 = self.module_71(output_module_66)
        output_module_71 = self.module_72(output_module_71)
        output_module_71 = self.module_73(output_module_71)
        output_module_71 = self.module_74(output_module_71)
        output_module_71 = self.module_75(input=output_module_71, other=output_module_66, alpha=1)
        output_module_76 = self.module_76(output_module_71)
        output_module_76 = self.module_77(output_module_76)
        output_module_76 = self.module_78(output_module_76)
        output_module_76 = self.module_79(output_module_76)
        output_module_76 = self.module_80(input=output_module_76, other=output_module_71, alpha=1)
        output_module_81 = self.module_81(output_module_76)
        output_module_81 = self.module_82(output_module_81)
        output_module_81 = self.module_83(output_module_81)
        output_module_81 = self.module_84(output_module_81)
        output_module_81 = self.module_85(input=output_module_81, other=output_module_76, alpha=1)
        output_module_86 = self.module_86(output_module_81)
        output_module_86 = self.module_87(output_module_86)
        output_module_86 = self.module_88(output_module_86)
        output_module_86 = self.module_89(output_module_86)
        output_module_86 = self.module_90(input=output_module_86, other=output_module_81, alpha=1)
        output_module_91 = self.module_91(output_module_86)
        output_module_91 = self.module_92(output_module_91)
        output_module_91 = self.module_93(output_module_91)
        output_module_91 = self.module_94(output_module_91)
        output_module_91 = self.module_95(input=output_module_91, other=output_module_86, alpha=1)
        output_module_96 = self.module_96(output_module_91)
        output_module_96 = self.module_97(output_module_96)
        output_module_96 = self.module_98(output_module_96)
        output_module_96 = self.module_99(output_module_96)
        output_module_96 = self.module_100(input=output_module_96, other=output_module_91, alpha=1)
        output_module_101 = self.module_101(output_module_96)
        output_module_101 = self.module_102(output_module_101)
        output_module_101 = self.module_103(output_module_101)
        output_module_101 = self.module_104(output_module_101)
        output_module_101 = self.module_105(input=output_module_101, other=output_module_96, alpha=1)
        output_module_106 = self.module_106(output_module_101)
        output_module_106 = self.module_107(output_module_106)
        output_module_108 = self.module_108(output_module_106)
        output_module_108 = self.module_109(output_module_108)
        output_module_108 = self.module_110(output_module_108)
        output_module_108 = self.module_111(output_module_108)
        output_module_108 = self.module_112(input=output_module_108, other=output_module_106, alpha=1)
        output_module_113 = self.module_113(output_module_108)
        output_module_113 = self.module_114(output_module_113)
        output_module_113 = self.module_115(output_module_113)
        output_module_113 = self.module_116(output_module_113)
        output_module_113 = self.module_117(input=output_module_113, other=output_module_108, alpha=1)
        output_module_118 = self.module_118(output_module_113)
        output_module_118 = self.module_119(output_module_118)
        output_module_118 = self.module_120(output_module_118)
        output_module_118 = self.module_121(output_module_118)
        output_module_118 = self.module_122(input=output_module_118, other=output_module_113, alpha=1)
        output_module_123 = self.module_123(output_module_118)
        output_module_123 = self.module_124(output_module_123)
        output_module_123 = self.module_125(output_module_123)
        output_module_123 = self.module_126(output_module_123)
        output_module_123 = self.module_127(input=output_module_123, other=output_module_118, alpha=1)
        output_module_123 = self.module_128(output_module_123)
        output_module_123 = self.module_129(output_module_123)
        output_module_123 = self.module_130(output_module_123)
        output_module_123 = self.module_131(output_module_123)
        output_module_123 = self.module_132(output_module_123)
        output_module_123 = self.module_133(output_module_123)
        output_module_123 = self.module_134(output_module_123)
        output_module_123 = self.module_135(output_module_123)
        output_module_123 = self.module_136(output_module_123)
        output_module_123 = self.module_137(output_module_123)
        output_module_138 = self.module_138(output_module_123)
        output_module_138 = self.module_139(output_module_138)
        output_module_138 = self.module_140(output_module_138)
        output_module_141 = self.module_141(output_module_123)
        output_module_141 = self.module_142(output_module_141)
        output_module_141 = self.module_143(input=output_module_141, size=None, scale_factor=[2.0,2.0], mode='nearest')
        output_module_141 = self.module_144(dim=1, tensors=[output_module_141,output_module_101])
        output_module_141 = self.module_145(output_module_141)
        output_module_141 = self.module_146(output_module_141)
        output_module_141 = self.module_147(output_module_141)
        output_module_141 = self.module_148(output_module_141)
        output_module_141 = self.module_149(output_module_141)
        output_module_141 = self.module_150(output_module_141)
        output_module_141 = self.module_151(output_module_141)
        output_module_141 = self.module_152(output_module_141)
        output_module_141 = self.module_153(output_module_141)
        output_module_141 = self.module_154(output_module_141)
        output_module_155 = self.module_155(output_module_141)
        output_module_155 = self.module_156(output_module_155)
        output_module_155 = self.module_157(output_module_155)
        output_module_158 = self.module_158(output_module_141)
        output_module_158 = self.module_159(output_module_158)
        output_module_158 = self.module_160(input=output_module_158, size=None, scale_factor=[2.0,2.0], mode='nearest')
        output_module_158 = self.module_161(dim=1, tensors=[output_module_158,output_module_59])
        output_module_158 = self.module_162(output_module_158)
        output_module_158 = self.module_163(output_module_158)
        output_module_158 = self.module_164(output_module_158)
        output_module_158 = self.module_165(output_module_158)
        output_module_158 = self.module_166(output_module_158)
        output_module_158 = self.module_167(output_module_158)
        output_module_158 = self.module_168(output_module_158)
        output_module_158 = self.module_169(output_module_158)
        output_module_158 = self.module_170(output_module_158)
        output_module_158 = self.module_171(output_module_158)
        output_module_158 = self.module_172(output_module_158)
        output_module_158 = self.module_173(output_module_158)
        output_module_158 = self.module_174(output_module_158)
        return (output_module_138,output_module_155,output_module_158)

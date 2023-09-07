""" 
  @Author: Zhiwen.Cai  
  @Date: 2022-09-15 15:23:45  
  @Last Modified by: Zhiwen.Cai  
  @Last Modified time: 2022-09-15 15:23:45  
"""
import torch
import torch.nn as nn
from models.convlstm import ConvLSTM
from models.spatial_branch import InputConv
from torchvision import models
from models.temporal_branch import DownConvBlocks,TemporalAggregator,UpSample,ConvlstmEncoder
resnet=models.resnet50(pretrained=True)
#resnet=models.resnet18(pretrained=True)

##  spatial encoder for high resolution imagries
class SpatialEncoder(nn.Module):
    def __init__(self,in_channels,bilinear=False):
        super().__init__()
        self.in_channels = in_channels

        self.bilinear = bilinear
     
        self.conv1 = InputConv(in_channels,64)            
        self.bn1=resnet.bn1
        self.relu=resnet.relu
        
        self.maxpool=resnet.maxpool       
        self.layer1=resnet.layer1
        self.layer2=resnet.layer2
        self.layer3=resnet.layer3
        self.layer4=resnet.layer4
        
    def forward(self,x):
        x = self.conv1(x)         # shape: (256,256) -> (128,128)
        x = self.bn1(x)
        # level0 = self.relu(self.bn1(level0))
        level1 = self.relu(x)     # shape: (64,128,128)
        level2 = self.maxpool(level1)    
        level2 = self.layer1(level2)   # shape: (256,64,64)
        level3 = self.layer2(level2)  # shape: (512,32,32)
        level4 = self.layer3(level3)  # shape: (1024,16,16),
        level5 = self.layer4(level4)  # shape: (2048,8,8)
        # print(level5.shape)
        return level1,level2,level3,level4,level5
    
## convlstm encoder for time series data 
class TemporalEncoder(nn.Module):
    def __init__(self,
                 in_channels,
                 kernel_size,
                 layer_num,
                 up_sample=True,
                 bidirectional=True,
                 ):
        super().__init__()
        filters=[64,256,512,1024,2048]
        self.up_sample = up_sample
        self.temporal_encoder = ConvlstmEncoder(in_channels,
                                                filters[0]//2,
                                                kernel_size,
                                                layer_num,
                                                bidirectional)
        self.temporal_aggregator = TemporalAggregator(filters[0]//2,
                                                     bidirectional)
        if up_sample:
            self.uplayer = UpSample(filters[0],filters[0],4)
            self.uplayer2 = UpSample(filters[1],filters[1],4)
            self.uplayer3 = UpSample(filters[2],filters[2],4)
            self.uplayer4 = UpSample(filters[3],filters[3],4)
            self.uplayer5 = UpSample(filters[4],filters[4],4)
            # self.uplayer = nn.Sequential(nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            #                                 nn.Conv2d(64,64,1,1,0))
        
        self.layer1 = DownConvBlocks(filters[0],filters[1],
                                     k=3,s=2,p=1,norm_type='batch')
        self.layer2 = DownConvBlocks(filters[1],filters[2],
                                     k=3,s=2,p=1,norm_type='batch')
        self.layer3 = DownConvBlocks(filters[2],filters[3],
                                     k=3,s=2,p=1,norm_type='batch')
        self.layer4 = DownConvBlocks(filters[3],filters[4],
                                     k=3,s=2,p=1,norm_type='batch')
        
    def forward(self,x):
        # x shapes: (8,3,32,32)
        te = self.temporal_encoder(x) # shapes: (8,64,32,32)
        level1,attn = self.temporal_aggregator(te) # shapes: (64,32,32)
        level2 = self.layer1(level1) # shapes: (256,16,16)
        level3 = self.layer2(level2) # shapes: (512,8,8)
        level4 = self.layer3(level3) # shapes: (1024,4,4)
        level5 = self.layer4(level4) # shapes: (2048,2,2)

        if self.up_sample:
            level1 = self.uplayer(level1) # shapes:(64,128,128)  
            level2 = self.uplayer2(level2) # shapes:(256,64,64)
            level3 = self.uplayer3(level3) # shapes:(512,32,32)
            level4 = self.uplayer4(level4) # shapes:(1024,16,16)
            level5 = self.uplayer5(level5) # shapes:(2048,8,8)

        return level1,level2,level3,level4,level5,attn
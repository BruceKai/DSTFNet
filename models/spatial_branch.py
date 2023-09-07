# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 21:00:05 2020

@author: d
"""
import torch.nn as nn
import torch
from models.attention import ChannelAttention,SpatialAttention
from torch.nn import functional as F

class InputConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        
        self.conv_3 = nn.Conv2d(in_channels, out_channels, 
                                    kernel_size=(3, 3), stride=(2, 2), 
                                    padding=(1, 1), bias=False)    
        self.conv_5 = nn.Conv2d(in_channels, out_channels, 
                                   kernel_size=(5, 5), stride=(2, 2), 
                                   padding=(2, 2), bias=False)
        self.conv_7 = nn.Conv2d(in_channels, out_channels, 
                                   kernel_size=(7, 7), stride=(2, 2), 
                                   padding=(3, 3), bias=False)

    def forward(self, x):
        conv_3_out = self.conv_3(x)
        conv_5_out = self.conv_5(x)
        conv_7_out = self.conv_7(x)
        x = conv_3_out+conv_5_out+conv_7_out
        return x
    
class ResBlock(nn.Module):

    def __init__(self,input_channels,output_channels):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(input_channels,output_channels,kernel_size=1,
                                            padding = 0),
                                  nn.BatchNorm2d(output_channels))
        self.res_conv = nn.Sequential(nn.Conv2d(output_channels,output_channels//2,kernel_size=3,
                                                 stride=1,padding=1,bias=True),
                                       nn.BatchNorm2d(output_channels//2),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(output_channels//2,output_channels,kernel_size=3,
                                                 stride=1, padding=1,bias=True),
                                       nn.BatchNorm2d(output_channels),
                                       )
        
    def forward(self,x):
        x = self.conv(x)
        # print(x.shape)
        res_x = self.res_conv(x)
        x = torch.relu(x+res_x)        
        return x   
 
class UpBlock(nn.Module):
    ''' refinement residual moduel
    '''
    def __init__(self,in_channels,out_channels,upsample_id):

        super().__init__()

        self.upsample_id = upsample_id
        if self.upsample_id == 1:
            self.cam = ContextAwareModule(in_channels,in_channels)
        else:
            self.scalefusion = MultiScaleFusion(in_channels)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.resblock = ResBlock(in_channels,out_channels)
    def forward(self,high_level,*kwargs):
        # x = self.conv(x)
        if self.upsample_id >1:
            low_level = kwargs[0]
            x, attn = self.scalefusion(high_level,low_level)
            out = self.upsample(x)
            out = self.resblock(out)
            return out, attn
        else:
            x = self.cam(high_level)
            out = self.upsample(x)
            out = self.resblock(out)
            return out


class MultiScaleFusion(nn.Module):
    def __init__(self,input_channels):
        super().__init__()
        output_channels = input_channels
        self.W_g = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(output_channels)
            )
        self.W_x = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(output_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(output_channels, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Sequential(nn.Conv2d(input_channels*2,
                                    output_channels,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    bias=True),
                            nn.BatchNorm2d(output_channels),
                            nn.ReLU(inplace=True))

    def forward(self,X_c,X_f):
        g1 = self.W_g(X_c)
        x1 = self.W_x(X_f)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        out = self.conv(torch.cat([X_c,X_f*psi],dim=1))
        return out,psi

    
class MultiBranchFusion(nn.Module):
    def __init__(self,input_channels,fuse_type):
        super().__init__()
        output_channels = input_channels
        self.fuse_type = fuse_type
        if self.fuse_type !='cat':
            self.spatial_attn = SpatialAttention()
            self.channel_attn = ChannelAttention(input_channels*2,input_channels*2,16)
        self.fuse_conv = nn.Sequential(nn.Conv2d(input_channels*2,
                                                 output_channels,
                                                 kernel_size=1,
                                                 stride=1,
                                                 padding=0,
                                                 bias=True),
                                       nn.BatchNorm2d(output_channels),
                                       nn.ReLU(inplace=True)) 
    def forward(self,spatial,temporal):
        if self.fuse_type =='cbam':
# CBAM block (Channel attn -> Spatial attn)
            x = torch.cat([spatial,temporal],dim = 1)
            channel_attn = self.channel_attn(x)
            out = x*channel_attn
            spatial_attn = self.spatial_attn(out)
            out = out*spatial_attn
            out = self.fuse_conv(out)    
        elif self.fuse_type =='cat':
            x = torch.cat([spatial,temporal],dim = 1)
            out = self.fuse_conv(x)
            channel_attn,spatial_attn = 0,0

        return out,channel_attn,spatial_attn

class OutConvExtent(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels=32
        self.out_channels = out_channels
        self.conv = nn.Sequential(nn.Conv2d(in_channels,mid_channels,kernel_size=3,padding=1),
                                nn.BatchNorm2d(mid_channels),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(mid_channels, out_channels, kernel_size=1)
                                ) 
    def forward(self, x):
        
        if self.out_channels > 1 :
            out = torch.softmax(self.conv(x),dim=1)
        else:
            out = torch.sigmoid(self.conv(x))
        return out

    
class OutConvDistance(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        mid_channels = in_channels//2
        self.conv = nn.Sequential(
                                nn.Conv2d(in_channels,mid_channels,kernel_size=3,padding=1),
                                nn.BatchNorm2d(mid_channels),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(mid_channels,mid_channels,kernel_size=3,padding=1),
                                nn.BatchNorm2d(mid_channels),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(mid_channels, 1, kernel_size=1),
                                      )

    def forward(self, x):
        out = torch.sigmoid(self.conv(x))
        return out


    
class OutConvBoundary(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        mid_channels = in_channels//2
        self.conv_v = nn.Sequential(
                        nn.Conv2d(in_channels,mid_channels,kernel_size=(3,1),padding=(1,0)) ,
                        nn.BatchNorm2d(mid_channels),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(mid_channels,mid_channels,kernel_size=(3,1),padding=(1,0)) ,
                        nn.BatchNorm2d(mid_channels),
                        nn.ReLU(inplace=True)  
                        )
        self.conv_h = nn.Sequential(
                        nn.Conv2d(in_channels,mid_channels,kernel_size=(1,3),padding=(0,1)) ,
                        nn.BatchNorm2d(mid_channels),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(mid_channels,mid_channels,kernel_size=(1,3),padding=(0,1)) ,
                        nn.BatchNorm2d(mid_channels),
                        nn.ReLU(inplace=True)  
                        )
        self.out_conv = nn.Conv2d(mid_channels, 1, kernel_size=1)

    def forward(self, x):
        x = self.conv_v(x) + self.conv_h(x)
        out = torch.sigmoid(self.out_conv(x))
        return out


class ContextAwareblock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,dilation,padding,pool_direct):
        super(ContextAwareblock,self).__init__()
        self.pool_direct = pool_direct # 0,1,2 represent no pooling,  horizontal,vertical
        self.kernel_size = kernel_size
        if self.pool_direct>0:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=1,padding=0),
                nn.ReLU(inplace=True) 
                )
        else: 
            self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,
                                dilation=dilation[0],padding=padding[0]
                                ) ,
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True) 
                        )
            self.conv2 = nn.Sequential(
                        nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,
                                dilation=dilation[1],padding=padding[1]
                                ) ,
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True) 
                        )
            self.conv3 = nn.Sequential(
                        nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,
                                dilation=dilation[2],padding=padding[2]
                                ) ,
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True) 
                        )
    def forward(self,x):
        if self.pool_direct>0:
            avg_ = torch.mean(x,dim=self.pool_direct+1,keepdim=True)
            x = self.conv(avg_)
        else:
            x = self.conv1(x) + self.conv2(x) + self.conv3(x)
        return x
    
class ContextAwareModule(nn.Module):
    def  __init__(self,in_channels,out_channels):
        super(ContextAwareModule,self).__init__()
        dilation = [1,2,4]
        self.cab1 = nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size=1,padding=0),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU(inplace=True))
        self.cab2 = ContextAwareblock(in_channels,out_channels//4,(1,3),dilation,[(0,1),(0,2),(0,4)],0)
        self.cab3 = ContextAwareblock(in_channels,out_channels//4,(3,1),dilation,[(1,0),(2,0),(4,0)],0)
        self.cab4 = ContextAwareblock(in_channels,out_channels//4,1,dilation,[0,0,0],1)
        self.cab5 = ContextAwareblock(in_channels,out_channels//4,1,dilation,[0,0,0],2)
        
        self.conv = nn.Sequential(nn.Conv2d(out_channels, out_channels, 1),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU(inplace=True))
    
    def forward(self,x):
        x1 = self.cab1(x)
        x2 = self.cab2(x)
        x3 = self.cab3(x)
        x4 = self.cab4(x)
        x4 = F.interpolate(x4,(x.shape[-2],x.shape[-1]),mode='bilinear')
        x5 = self.cab4(x)
        x5 = F.interpolate(x5,(x.shape[-2],x.shape[-1]),mode='bilinear')
        out = self.conv(x1+torch.cat([x2,x3,x4,x5],dim=1))
        return out

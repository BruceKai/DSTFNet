import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import Softmax

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Sequential(nn.Conv2d(2,1,kernel_size=1,
                                          stride=1,padding=0,bias=True),
                                nn.BatchNorm2d(1),
                                nn.Sigmoid())
    def forward(self,x):
        avg_ = torch.mean(x,dim=1,keepdim=True)
        max_,_ = torch.max(x,dim=1,keepdim=True)
        x = torch.cat([avg_,max_],dim=1)
        spatial_attn = self.conv(x)
        return spatial_attn

class ChannelAttention(nn.Module):
    def __init__(self,input_channels,output_channels,reduction):
        super().__init__()
        # output_channels = input_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.channel_attn_block=nn.Sequential( 
                                        nn.Conv2d(input_channels,input_channels//reduction,
                                            kernel_size=(1,1),stride=1,padding=0,bias=True),
                                        # nn.BatchNorm2d(input_channels//reduction),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(input_channels//reduction,output_channels,
                                               kernel_size=(1,1),stride=1,padding=0,bias=True),
                                        # nn.BatchNorm2d(input_channels),
                                        )
        
    def forward(self,x):
        max_pool_re = self.max_pool(x)
        avg_pool_re = self.avg_pool(x)           
        max_channel_attn = self.channel_attn_block(max_pool_re)
        avg_channel_attn = self.channel_attn_block(avg_pool_re)
        channel_attn = torch.sigmoid(max_channel_attn+avg_channel_attn)
        
        return channel_attn


def INF(B,H,W):
     return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H),0).unsqueeze(0).repeat(B*W,1,1)

class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, in_dim):
        super(CrissCrossAttention,self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//32, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//32, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        
        m_batchsize, _, height, width = x.size()
        
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)

        return self.gamma*(out_H + out_W)


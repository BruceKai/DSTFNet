import torch
from models.convlstm import ConvLSTM
from torch import nn

# DownConvBlocks for SITS
class DownConvBlocks(nn.Module):
    def __init__(self,in_channels,out_channels,k,s,p,
                 n_groups=4,norm_type='batch'):
        super().__init__()
        # self.norm_type = norm_type
        norm = dict({'batch':nn.BatchNorm2d,
                     'instance':nn.InstanceNorm2d,
                     'group':lambda num_feats: nn.GroupNorm(
                                num_channels=num_feats,
                                num_groups=n_groups)
                     })

        self.down = nn.Sequential(nn.Conv2d(in_channels, in_channels, 
                                            kernel_size=k,padding=p,stride=s),
                                  norm[norm_type](in_channels),
                                  nn.ReLU(inplace=True))
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 
                                            kernel_size=3,padding=1,stride=1),
                                  norm[norm_type](out_channels),
                                  nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 
                                            kernel_size=3,padding=1,stride=1),
                                  norm[norm_type](out_channels),
                                  nn.ReLU(inplace=True))
    def forward(self,x):
        # b,t,c,h,w = x.shape
        # if self.norm_type == 'group':
        #     x = x.reshape(b*t,c,h,w)
        down = self.down(x)
        out = self.conv1(down)
        out = out+self.conv2(out)
        return out

class TemporalAggregator(nn.Module):
    def __init__(self,hidden_size,bidirectional=True):
        super().__init__()

        self.conv = nn.Sequential(nn.Conv2d(in_channels=(1*bidirectional+1)*hidden_size,
                                    out_channels=1,kernel_size=1,
                                    stride=1,padding=0,bias=True),
                                  nn.BatchNorm2d(1))
                  
    def forward(self,x):

        b,t,c,h,w = x.shape # input batch_size,seq_len,channel_size
        temporal_attn = self.conv(x.reshape(b*t,c,h,w))
        temporal_attn = torch.softmax(temporal_attn.reshape(b,t,1,h,w),dim=1)
        weighted_out = torch.sum(temporal_attn*x,dim=1)
            
        return weighted_out,temporal_attn


class UpSample(nn.Module):
    def __init__(self,in_channels,out_channels,up_scale):
        super().__init__()
        self.uplayer = nn.Sequential(nn.Upsample(scale_factor=up_scale, mode='bilinear', align_corners=True),
                                    nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1,stride=1),
                                    nn.BatchNorm2d(out_channels),
                                    nn.ReLU())
    def forward(self,x):
        shapes = x.shape 
        if len(shapes)>4:
            b,t,c,h,w = shapes
            x = x.reshape(b*t,c,h,w)
        out = self.uplayer(x)
        return out

class ConvlstmEncoder(nn.Module):
    def __init__(self,in_channels,hidden_size,kernel_size,
                 layer_num,bidirectional=True,**kwargs):
        super().__init__()
        kernel_size = (kernel_size,kernel_size)
        
        self.forward_rnn = ConvLSTM(in_channels,hidden_size,
                                kernel_size,layer_num, True, True, False)
        self.bidirectional = bidirectional
        if self.bidirectional:
            self.backward_rnn = ConvLSTM(in_channels,hidden_size,
                                    kernel_size,layer_num, True, True, False)
    def forward(self,x_forward):
        
        forward_out,_ = self.forward_rnn(x_forward)
        forward_out = forward_out[0]
        if self.bidirectional:            
            x_backward = torch.flip(x_forward,dims=[1])
            backward_out,_ = self.backward_rnn(x_backward)
            backward_out = torch.flip(backward_out[0],dims = [1])
            convlstm_out = torch.cat([forward_out,backward_out],dim=2)
        else:
            convlstm_out = forward_out
        return convlstm_out



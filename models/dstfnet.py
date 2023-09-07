import torch
import torch.nn as nn
from models.encoder import SpatialEncoder as SE,\
                        TemporalEncoder as TE
                        
from models.decoder import Decoder


class DSTFNet(nn.Module):
    def __init__(self,in_channels,num_classes,in_features,
                 layer_num,mode,fuse_type_b,**kwargs):
        super(DSTFNet, self).__init__()
        filters=[64,256,512,1024,2048]
        self.mode = mode
        if mode=='dual': # Dual branch model
            self.spatial_encoder = SE(in_channels)
            self.temporal_encoder = TE(in_features,3,layer_num)
            self.decoder = Decoder(mode,True,num_classes,fuse_type_b)
            
        elif mode=='temporal': # temporal branch model
            self.temporal_encoder = TE(in_features,3,layer_num,False)
            self.decoder = Decoder(mode,False,num_classes,fuse_type_b)
  
        elif (self.mode=='spatial')|(self.mode=='early fusion'): # spatial branch model
            self.spatial_encoder = SE(in_channels)
            self.decoder = Decoder(mode,True,num_classes,fuse_type_b)
    
    def forward(self,*kwargs):

        if self.mode=='dual': 
            spatial_data = kwargs[0]
            temporal_data = kwargs[1]
            sp_features = self.spatial_encoder(spatial_data)
            tm_features = self.temporal_encoder(temporal_data)
            prediction = self.decoder(sp_features,tm_features[:-1])
            
        elif self.mode=='temporal': 
            temporal_data = kwargs[0]
            tm_features = self.temporal_encoder(temporal_data)
            prediction = self.decoder(tm_features[:-1])

        elif (self.mode=='spatial')|(self.mode=='early fusion'): #
            spatial_data = kwargs[0]
            sp_features = self.spatial_encoder(spatial_data) 
            prediction = self.decoder(sp_features) 
        prediction.update({'tattn':tm_features[-1]})
        return prediction

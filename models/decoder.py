import torch
import torch.nn as nn
from torch.nn import functional as F    
from models.spatial_branch import UpBlock,MultiBranchFusion   
from models.spatial_branch import OutConvExtent,OutConvDistance,OutConvBoundary,ContextAwareModule
class Decoder(nn.Module):
    def __init__(self,
                 mode,
                 up_sample,
                 num_classes,
                 fuse_type_b,
                 **kwargs):
        super().__init__()
        filters=[64,256,512,1024,2048]
        self.filters = filters
        self.mode = mode
        self.up_sample = up_sample 
        self.upblock_1 = UpBlock(filters[4],filters[3],1)  
        self.upblock_2 = UpBlock(filters[3],filters[2],2)        
        self.upblock_3 = UpBlock(filters[2],filters[1],3)
        self.upblock_4 = UpBlock(filters[1],filters[0],4)
        if mode =='dual':
            self.branchfusion = SpatialTemporalFusion(fuse_type_b)
            self.upblock_5 = UpBlock(filters[0],filters[0],5)
        elif (mode =='spatial')|(mode=='early fusion'):
            self.upblock_5 = UpBlock(filters[0],filters[0],5)

        elif (mode=='temporal') & (up_sample):
            self.upblock_5 = UpBlock(filters[0],filters[0],5)
        else:  
            pass
        self.semantic_seg_block = SemanticSeg(num_classes=num_classes,filters=filters)

        
    def forward(self,*wargs):
        
        if self.mode=='dual':
            spatial = wargs[0]
            temporal = wargs[1]
            fusion1,fusion2,fusion3,fusion4,fusion5,catt,satt = self.branchfusion(spatial,temporal)
            out = self.upblock_1(fusion5)
            out,attn1 = self.upblock_2(out,fusion4)
            out,attn2 = self.upblock_3(out,fusion3)
            out,attn3 = self.upblock_4(out,fusion2)
            out,attn4 = self.upblock_5(out,fusion1)
        elif (self.mode == 'spatial')|(self.mode=='early fusion'):
            spatial = wargs[0]
            sp_l1,sp_l2,sp_l3,sp_l4,sp_l5 = spatial
            out = self.upblock_1(sp_l5)
            out,_ = self.upblock_2(out,sp_l4)
            out,_ = self.upblock_3(out,sp_l3)
            out,_ = self.upblock_4(out,sp_l2)
            out,_ = self.upblock_5(out,sp_l1)
        elif (self.mode=='temporal') & (self.up_sample):
            temporal = wargs[0]
            tm_l1,tm_l2,tm_l3,tm_l4,tm_l5 = temporal
            out = self.upblock_1(tm_l5)
            out,_ = self.upblock_2(out,tm_l4)
            out,_ = self.upblock_3(out,tm_l3)
            out,_ = self.upblock_4(out,tm_l2)
            out,_ = self.upblock_5(out,tm_l1) 
        else:
            temporal = wargs[0]
            tm_l1,tm_l2,tm_l3,tm_l4,tm_l5 = temporal
            out = self.upblock_1(tm_l5)
            out,_ = self.upblock_2(out,tm_l4)
            out,_ = self.upblock_3(out,tm_l3)
            out,_ = self.upblock_4(out,tm_l2)
            out = out+tm_l1    
        prediction = self.semantic_seg_block(out)
        prediction.update({'satt':satt,
                           'catt':catt,
                           'att':[attn1,attn2,attn3,attn4]})
        return prediction


class SpatialTemporalFusion(nn.Module):
    def __init__(self, fuse_type_b):
        super(SpatialTemporalFusion, self).__init__()
        filters=[64,256,512,1024,2048]
        
        self.branchfusion1 = MultiBranchFusion(filters[0],fuse_type_b)
        self.branchfusion2 = MultiBranchFusion(filters[1],fuse_type_b)
        self.branchfusion3 = MultiBranchFusion(filters[2],fuse_type_b)                     
        self.branchfusion4 = MultiBranchFusion(filters[3],fuse_type_b) 
        self.branchfusion5 = MultiBranchFusion(filters[4],fuse_type_b)
         
    def forward(self,sp_features,tm_features):
        sp_l1,sp_l2,sp_l3,sp_l4,sp_l5 = sp_features
        tm_l1,tm_l2,tm_l3,tm_l4,tm_l5 = tm_features
        
        fusion1,catt1,satt1= self.branchfusion1(sp_l1,tm_l1)
        fusion2,catt2,satt2= self.branchfusion2(sp_l2,tm_l2)
        fusion3,catt3,satt3= self.branchfusion3(sp_l3,tm_l3)                     
        fusion4,catt4,satt4= self.branchfusion4(sp_l4,tm_l4)
        fusion5,catt5,satt5= self.branchfusion5(sp_l5,tm_l5)
        
        return fusion1,fusion2,fusion3,fusion4,fusion5,[catt1,catt2,catt3,catt4,catt5],\
            [satt1,satt2,satt3,satt4,satt5]



class SemanticSeg(nn.Module):
    def __init__(self,
                 num_classes,
                 filters
                 ):
        super(SemanticSeg, self).__init__()
        

        self.extent_conv = OutConvExtent(filters[0],num_classes)
        self.distance_conv = OutConvDistance(filters[0])
        self.boundary_conv = OutConvBoundary(filters[0])

        
    def forward(self,x):
        
        distance = self.distance_conv(x)

        extent = self.extent_conv(x)
        boundary = self.boundary_conv(x)
        prediction = {'extent':extent,
                       'distance':distance,
                       'boundary':boundary}
            
        return prediction
    

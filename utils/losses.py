import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

from utils.focalloss import FocalLoss
from utils.centerloss import CenterLoss
from utils.diceloss import BinaryTanimotoLoss 

def get_semantic_metric(pred,ref):
    """
    Args:
        pred([array]): the prediction of model
        ref([array]): the reference data
    """
    b,h,w = ref.squeeze(1).shape
    ref_size = b*h*w
    
    pred = pred.squeeze(1)  
    pred = (pred>=0.5).float()
    f_score = (2*(pred*ref).sum()+1e-6)/((pred+ref).sum()+1e-6)
    tp,tn,fp,fn = (pred*ref).sum()/ref_size, ((1-pred)*(1-ref)).sum()/ref_size,\
                (pred*(1-ref)).sum()/ref_size,((1-pred)*ref).sum()/ref_size
    mcc = ((tp*tn-fp*fn)+1e-6)/(torch.sqrt((tp+fn)*(tp+fp)*(tn+fp)*(tn+fn))+1e-6)
    
    return f_score,mcc

class Loss(nn.Module):
    def __init__(self,reduction='mean'):
        super(Loss, self).__init__()
        
        self.binary_threshold = 0.5
        self.semantic_seg_loss = BinaryTanimotoLoss(reduction=reduction)
        # self.distance_loss = nn.MSELoss(reduction=reduction)
        self.value = []

    
    def forward(self,predict,target,w):
        
        loss_extent = self.semantic_seg_loss(predict['extent'],target[:,0,:,:])
        loss_boundary = self.semantic_seg_loss(predict['boundary'],target[:,1,:,:])
        loss_distance = self.semantic_seg_loss(predict['distance'],target[:,2,:,:])
   
        
        f_score,mcc = get_semantic_metric(predict['extent'],target[:,0,:,:])
        edge_f,_ = get_semantic_metric(predict['boundary'],target[:,1,:,:])

        loss = w[0]*loss_extent+w[1]*loss_boundary+w[2]*loss_distance
        self.value = [
                    loss_extent.item()/target.shape[0],
                    loss_boundary.item()/target.shape[0],
                    loss_distance.item()/target.shape[0],
                    f_score.detach().cpu().numpy(),
                    mcc.detach().cpu().numpy(),
                    edge_f.detach().cpu().numpy(),
                    ]   
        return loss
            

    


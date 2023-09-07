""" 
  @Author: Zhiwen.Cai  
  @Date: 2022-09-15 14:56:16  
  @Last Modified by: Zhiwen.Cai  
  @Last Modified time: 2022-09-15 14:56:16  
"""
import copy
import logging

import numpy as np
import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import OrderedDict
from utils import losses
from datasets import mydataset
from torch.optim.lr_scheduler import CosineAnnealingLR,LambdaLR,MultiStepLR,ReduceLROnPlateau
from torch.cuda.amp import autocast as autocast
import datetime
# % validate the module accuracy
def evaluate(val_data,net,loss,devices,w):

    val_loss = []
    net.eval()
    with torch.no_grad():
        for element in val_data:  
            target = element[-1]
            target = target.to(devices,dtype=torch.float)
            # [v.to(devices) for k,v in target]
            X = [x.to(devices,dtype=torch.float) for x in element[:-1]]
            with autocast():
                pred = net(*X)

                l = loss(pred,target,w)

            val_loss.append(loss.value)
    val_loss = np.array(val_loss).mean(0)

    return  val_loss

# % define get_logger
def get_logger(filename, mode, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "%(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, mode)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

# %Train model
def train_model(net,
                devices,
                batchsize,
                lr,
                num_classes,
                max_epoch,
                train_folder,
                val_folder,
                num_workers,
                data_aug,
                model_name,
                mode,
                resume
                ):
    """
        model training
    Args:
        net ([torch model]): the deep neural network
        devices ([torch.device]): 
        batchsize ([int]): batch size for training
        lr ([float]): learning rate for training
        num_classes ([int]): the number of classes
        max_epoch ([int]): the maximum number of epochs
        train_folder ([str]): the folder containing training data
        val_folder ([str]): the folder containing validation data
        num_workers ([int]): the number of workers
        data_aug ([bool]): data augmentation
        model_name ([str]): the file name of the networks
        mode ([str]): the multi modality
        resume([bool]): resume
    """
 
    loss = losses.Loss(reduction='sum')

    start_epoch = 1
    trainer = torch.optim.Adam([{'params':net.parameters(),'initial_lr':lr}],lr=lr,weight_decay=1e-5)
    scheduler = MultiStepLR(trainer, milestones=[15,20,30,60], gamma=0.3)
    log_path = './logging/'+model_name+'.log'

    if resume:
        check_point_path = r'./result/model/'+model_name+'_optimal.pth'
        check_point = torch.load(check_point_path)
        net.load_state_dict(check_point['model_state_dict'])
        trainer.load_state_dict(check_point['optimizer_state_dict'])
        scheduler.load_state_dict(check_point['lr_state_dict'])
        start_epoch = check_point['epoch']+1
        logger = get_logger(log_path,"a")
        loss_information = np.load(r'./result/loss/'+model_name+'_optimal.npy', allow_pickle=True)
        loss_information = loss_information[()]
        mcc_max = loss_information['val_loss'][-1][-2]
        l_hat_1,l_hat_2 = check_point['l_hat']
    else:
        logger = get_logger(log_path,"w")
        logger.info('\r start training!')
        logger.info('\r config params')
        logger.info('\r model name: {}'.format(model_name))
        logger.info('\r batch size: {}'.format(batchsize))
        logger.info('\r mode: {}'.format(mode))
        logger.info('\r initial learning rate: {}'.format(lr))
        logger.info('\r num classes: {}'.format(num_classes))
        logger.info('\r date time: {}'.format(datetime.datetime.now()))
        
        loss_information = dict({'train_loss':[],'val_loss':[],'epoch':[]})
        mcc_max = 0 
        l_hat_2 = torch.ones(3)
        l_hat_1 = torch.ones(3)


    for epoch in range(start_epoch,max_epoch+1):
       
        net.train()
        if mode =='dual':
            train_dataset = mydataset.DualDataset(train_folder,data_aug=data_aug)
            val_dataset = mydataset.DualDataset(val_folder)
        elif mode == 'temporal':
            train_dataset = mydataset.TemporalDataset(train_folder,data_aug=data_aug)
            val_dataset=mydataset.TemporalDataset(val_folder)
        elif (mode == 'spatial')|(mode=='early fusion'):
            early_fusion = True if mode=='early fusion' else False
            train_dataset = mydataset.SpatialDataset(train_folder,data_aug=data_aug,early_fusion=early_fusion)
            val_dataset = mydataset.SpatialDataset(val_folder,early_fusion=early_fusion)
            
        train_kwargs = dict({'dataset':train_dataset,
                             'batch_size':batchsize,
                             'shuffle':True,
                             'num_workers':num_workers,
                             'pin_memory':True,})
        train_data = DataLoader(**train_kwargs)
         
        val_kwargs = dict({'dataset':val_dataset,
                             'batch_size':batchsize,
                             'shuffle':True,
                             'num_workers':num_workers,
                             'pin_memory':True,})
        val_data = DataLoader(**val_kwargs)                                              
       
        j=1
        extent_loss, boundary_loss, dist_loss = 0.,0.,0.
        w_hat = l_hat_1/l_hat_2
        w = 1*torch.exp(w_hat)/torch.exp(w_hat).sum()
        logger.info(w.cpu().detach().numpy())
        w = w.to(devices)
        tmp_loss = []
        with tqdm(iterable=train_data,desc=f'Epoch {epoch}/{max_epoch}', unit='batch')as pbar:

            for element in train_data:  

                trainer.zero_grad()  
                type_Y = torch.float 
                target = element[-1]
                target = target.to(devices,dtype=torch.float)
                # [v.to(devices) for k,v in target]
                # X = element[:-1]
                X = [x.to(devices,dtype=torch.float) for x in element[:-1]]
                with autocast():
                    # print(len(X))
                    pred = net(*X)                   
                    l = loss(pred,target,w)

                l.backward()
                trainer.step() 
                tmp_loss.append(loss.value)

                extent_loss += tmp_loss[j-1][0]
                boundary_loss += tmp_loss[j-1][1]
                dist_loss += tmp_loss[j-1][2]
    
                j += 1    
                pbar.set_postfix( 
                    loss1 = extent_loss/j,
                    loss2 = boundary_loss/j,
                    loss3 = dist_loss/j,
                    )
                pbar.update()
        
        # reset weights and learning rate 
        scheduler.step()
        if epoch == 1:
            l_hat_2[0],l_hat_2[1],l_hat_2[2]  = extent_loss/j,boundary_loss/j,dist_loss/j
            l_hat_1[0],l_hat_1[1],l_hat_1[2]  = extent_loss/j,boundary_loss/j,dist_loss/j
        else:
            l_hat_2 = copy.deepcopy(l_hat_1)
            l_hat_1[0],l_hat_1[1],l_hat_1[2]=  extent_loss/j,boundary_loss/j,dist_loss/j
        tmp_loss = np.array(tmp_loss).mean(0)
        # if (epoch == 1) | (epoch % 2 == 0):
        val_loss = evaluate(val_data,net,loss,devices,w)
        info1 = ' mode\t loss1\t loss2\t loss3\t fscore\t mcc\t edge-f\t' 
        info2 = ' train\t %.3f\t %.3f\t %.3f\t %.3f\t %.3f\t %.3f'%(
                    tmp_loss[0],tmp_loss[2],tmp_loss[1],tmp_loss[-3],tmp_loss[-2],tmp_loss[-1]) 
        info3 = ' val\t %.3f\t %.3f\t %.3f\t %.3f\t %.3f\t %.3f'%(
                    val_loss[0],val_loss[2],val_loss[1],val_loss[-3],val_loss[-2],val_loss[-1])

        logger.info('\r Epoch {}/{}:'.format(epoch,max_epoch))       
        logger.info(info1)     
        logger.info(info2)
        logger.info(info3)   
        
        loss_information['train_loss'].append(tmp_loss)
        loss_information['val_loss'].append(val_loss)
        loss_information['epoch'].append([epoch])

        if (val_loss[-2] > mcc_max) & (val_loss[-2]<1):
            mcc_max = val_loss[-2]
            check_point = dict({'model_state_dict':net.state_dict(),
                               'optimizer_state_dict':trainer.state_dict(),
                               'lr_state_dict': scheduler.state_dict(),
                               'l_hat':[l_hat_1,l_hat_2],
                               'epoch':epoch}) 
            optimal_model_path = './result/model/'+model_name+'_optimal.pth'
            optimal_loss_path = './result/loss/'+model_name+'_optimal.npy'
            torch.save(check_point,optimal_model_path)
            np.save(optimal_loss_path,np.array(loss_information))

    state_dict = dict({'model_state_dict':net.state_dict(),
                        'epoch':epoch}) 
    model_path = './result/'+model_name+'.pth'
    loss_path = './result/'+model_name+'.npy'
    torch.save(state_dict,model_path)
    np.save(loss_path,np.array(loss_information))
    logger.info('\r finish training')
    logger.info('\r date time: {}'.format(datetime.datetime.now()))
    logger=[]
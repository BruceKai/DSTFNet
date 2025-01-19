
import torch
from collections import OrderedDict
from utils import train
from utils.BalanceDataparallel import BalancedDataParallel
from models import dstfnet
import os


batch_size = 64
lr = 3e-4
GPU0_BSZ = 32
MAX_EPOCH = 120
NUM_WORKERS = 6
ACC_GRAD = 1

NUM_CLASSES = 1
INPUT_FEATURES = 4 # 4 spectral bands 
HIDDEN_SIZES = 128
NUM_LAYERS = 2

FUSE_TYPE_B = 'cbam' 
MODE = 'dual' # 'spatial', 'temporal'

model_name = 'dstfnet_dual_cbam'

IN_CHANNELS = 8*4+4 if MODE =='early fusion' else 4 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = dstfnet.DSTFNet(
                    IN_CHANNELS,  # number of input bands of single date VHR image
                    NUM_CLASSES,  
                    INPUT_FEATURES, # number of input bands of MRSITS 
                    NUM_LAYERS,
                    MODE,
                    FUSE_TYPE_B,
                  )

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
# dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = BalancedDataParallel(GPU0_BSZ // ACC_GRAD,model,device_ids=[0,1],output_device=0)

train_folder = r'**/train'
val_folder = r'**/val'

model = model.to(device)
train_kwargs = dict({'net':model,
                    'devices':device,
                    'batchsize':batch_size,
                    'lr':lr,
                    'num_classes':NUM_CLASSES,
                    'max_epoch':MAX_EPOCH,
                    'train_folder':train_folder,
                    'val_folder':val_folder,
                    'num_workers':NUM_WORKERS,
                    'model_name':model_name,
                    'data_aug':True,
                    'mode':MODE,
                    'resume':False
                    })
train.train_model(**train_kwargs)

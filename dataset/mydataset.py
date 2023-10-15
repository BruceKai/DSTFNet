import glob

import numpy as np
import rasterio as rio
import torch
from rasterio.enums import Resampling
from torch.utils.data import Dataset
from torchvision.transforms import Normalize
from skimage import  morphology as mp
from datasets.data_augmentation import DataAug
from skimage.transform import rescale

transform = DataAug()

def image_linear_transform(image, scale=2):
    image = np.float32(image)
    
    for i in range(image.shape[0]):
        min_,max_ = np.percentile(image[i],[scale,100-scale])
        image[i] = (image[i]-min_)/(max_-min_)
    return np.clip(image,0,1)
        
class DualDataset(Dataset):
    def __init__(self,root_dir,data_aug=False):
        super(DualDataset, self).__init__()
        img_s2_path=glob.glob(root_dir+'/img_s2/*.tif')
        img_gf2_path=glob.glob(root_dir+'/img_gf2/*.tif')
        lbl_path=glob.glob(root_dir+'/lbl/*.tif')

        img_s2_path.sort()
        img_gf2_path.sort()
        lbl_path.sort()
        
        self.img_s2_path = img_s2_path
        self.img_gf2_path = img_gf2_path
        self.lbl_path = lbl_path
        self.data_aug = data_aug

    def __len__(self):
        return len(self.img_gf2_path)

    def __getitem__(self, index):
        
        img_gf2 = rio.open(self.img_gf2_path[index])
        img_s2 = rio.open(self.img_s2_path[index])
        lbl = rio.open(self.lbl_path[index])

        img_s2 = img_s2.read()
        # img_s2 = img_s2.read()
        img_s2 = img_s2/10000.
        img_s2[np.isnan(img_s2)]=0  
         
        
        img_gf2=img_gf2.read()

        img_gf2[(img_gf2<0)|(img_gf2>10000)]=0
        img_gf2=img_gf2/10000.
        scale = np.random.randint(1,4)
        img_gf2 = image_linear_transform(img_gf2,2)
        img_gf2[np.isnan(img_gf2)]=0
        
        # extent,boundary,distance
        target = lbl.read() 

        target[1] = mp.dilation(target[1],np.ones([3,3]))

        target = np.float32(target)
        target[(target>10000)|(target<-1000)]=0
        target[2] = target[2]/10000.
        
        # print(target.shape)
        if self.data_aug:
            input_data = dict({'image':[img_gf2,img_s2],
                               'label':target})
            input_data = transform.transform(input_data)
            img_gf2 = input_data['image'][0].copy()
            img_s2 = input_data['image'][1].copy()
            target = input_data['label'].copy() 

        # (t*c,h,w)->(t,c,h,w)       
        img_s2 = img_s2.reshape(8,4,img_s2.shape[1],img_s2.shape[2])
            
        return torch.from_numpy(img_gf2),torch.from_numpy(img_s2),torch.from_numpy(target)
    

class SpatialDataset(Dataset):
    def __init__(self,root_dir,data_aug=False,early_fusion=False):
        super(SpatialDataset, self).__init__()
        
        img_s2_path=glob.glob(root_dir+'/img_s2/*.tif')
        img_gf2_path=glob.glob(root_dir+'/img_gf2/*.tif')
        lbl_path=glob.glob(root_dir+'/lbl/*.tif')
        
        img_s2_path.sort()
        img_gf2_path.sort()
        lbl_path.sort()

        self.img_s2_path = img_s2_path
        self.img_gf2_path = img_gf2_path
        self.lbl_path = lbl_path
        self.data_aug = data_aug
        self.early_fusion = early_fusion
    
    def __len__(self):
        return len(self.img_gf2_path)

    def __getitem__(self, index):
        
        img_gf2 = rio.open(self.img_gf2_path[index])
        lbl = rio.open(self.lbl_path[index])
        img_gf2 = img_gf2.read()
        img_gf2[(img_gf2<0)|(img_gf2>10000)]=0
        img_gf2=img_gf2/10000.
        img_gf2 = image_linear_transform(img_gf2)
        img_gf2[np.isnan(img_gf2)]=0

        # extent,boundary,distance
        target = lbl.read() 
        target[1] = mp.dilation(target[1],np.ones([3,3]))
        target = np.float32(target)
        target[(target>10000)|(target<-1000)]=0
        target[2] = target[2]/10000.
        
        
        if self.early_fusion:
            img_s2 = rio.open(self.img_s2_path[index])
            img_s2 = img_s2.read(out_shape=(img_s2.count,img_gf2.shape[-1],img_gf2.shape[-1]),
                                 resampling=Resampling.bilinear)
            img_s2 = img_s2/10000.
            img_s2[np.isnan(img_s2)]=0
            img_gf2 = np.concatenate([img_gf2,img_s2],axis=0)
        
        if self.data_aug:
            input_data = dict({'image':[img_gf2],
                               'label':target})
            input_data = transform.transform(input_data)
            img_gf2 = input_data['image'][0].copy()
            target = input_data['label'].copy()
        
        return torch.from_numpy(img_gf2),torch.from_numpy(target)

class TemporalDataset(Dataset):
    def __init__(self,root_dir,data_aug=False):
        super(TemporalDataset, self).__init__()
        
        img_s2_path=glob.glob(root_dir+'/img_s2/*.tif')
        lbl_path=glob.glob(root_dir+'/lbl/*.tif')
        
        img_s2_path.sort()
        lbl_path.sort()

        self.img_s2_path = img_s2_path
        self.lbl_path = lbl_path
        self.data_aug = data_aug
    
    def __len__(self):
        return len(self.img_s2_path)

    def __getitem__(self, index):
        
        img_s2 = rio.open(self.img_s2_path[index])
        lbl = rio.open(self.lbl_path[index])

        img_s2 = img_s2.read()
        img_s2 = img_s2/10000.
        img_s2[np.isnan(img_s2)]=0
        
        

        # extent,boundary,distance
        target = lbl.read() 
        target[1] = mp.dilation(target[1],np.ones([3,3]))
        target = np.float32(target)
        target[(target>10000)|(target<-1000)]=0
        target[2] = target[2]/10000.
        
        
        scale = lbl.width//img_s2.shape[-1]
        params = {
                'image':target.transpose(1,2,0),
                  'scale':1/scale,
                  'mode':'constant',
                  'clip':True,
                  'preserve_range':True,
                  'multichannel':True
                  }
        
        target = rescale(**params)
        target = target.transpose(2,0,1)
        
        if self.data_aug:
            input_data = dict({'image':[img_s2],
                               'label':target})
            input_data = transform.transform(input_data)
            img_s2 = input_data['image'][0].copy()
            target = input_data['label'].copy()
            
        img_s2 = img_s2.reshape(8,4,img_s2.shape[1],img_s2.shape[2])
        
        return torch.from_numpy(img_s2),torch.from_numpy(target)

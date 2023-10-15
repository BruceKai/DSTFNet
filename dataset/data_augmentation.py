import numpy as np
import random

def random_rotate(image,angles):
    angle = random.choice(angles)
    if isinstance(image,list):
        for i,img in enumerate(image):
            axes = (0,1) if img.ndim <3 else (1,2)
            image[i] = np.rot90(img,angle//90,axes=axes)
    return image

def random_filp(image):
    p = random.random()
    if isinstance(image,list):
        for i,img in enumerate(image):      
            if p >0.5:
                axis = 0 if img.ndim<3 else 1 
                image[i] = np.flip(img,axis=axis)
            else:
                axis = 1 if img.ndim<3 else 2 
                image[i] = np.flip(img,axis=axis)
    return image

class DataAug(object):
    def __init__(self):
        self.angles = [90,180,270]
    def transform(self,input_data):
        proba = random.random()
        image = input_data['image']
        label = input_data['label']
        
        image.extend([label])
        if proba <0.4:
            image = random_filp(image)
        else:
            image = random_rotate(image,self.angles)
        
        input_data.update({'image':image[0:-1],
                           'label':image[-1]})
        return input_data

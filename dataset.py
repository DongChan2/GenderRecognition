#%%
import torch
import torch.nn as nn 
import torch.optim as optim 
import os 
import sys
import numpy as np 
import cv2

from torch.utils.data import DataLoader,Dataset
import torch.multiprocessing as mp
import torchvision.transforms as transforms

import random
from tqdm import tqdm

import glob


    
import time    

  
def gender_labeling(path):
    if (path.split("/")[-1]).split("_")[0] == 'male':
        #return torch.tensor(0)
        return torch.tensor(0)
              
    elif (path.split("/")[-1]).split("_")[0] == 'female':
        #return torch.tensor(1)
        return torch.tensor(1)


class RegDB(Dataset):
    def __init__(self,root_path,mode='train',transform=None, K=1):
        super().__init__()
        print("\n---[ RegDB init ]---\n")
        
        self.mp_method = mp.get_start_method()
        print("Detected multi-processing start method:", self.mp_method)  # spawn, fork
        
        if self.mp_method == "spawn":
            self._pack   = None
            self._unpack = None
        elif self.mp_method == "fork":
            self._pack   = transforms.ToTensor()
            self._unpack = transforms.ToPILImage()
        
        self.K = K 
        
        if self.K == 1:
            if mode=='train':
                self.root_path = root_path+"/RegDB/subset1/Thermal Images/train"
            elif mode=='validation':
                self.root_path = root_path+"/RegDB/subset1/Thermal Images/valid"
            else:
                self.root_path = root_path+"/RegDB/subset1/Thermal Images/test"
                
                
        elif self.K ==2:
            if mode=='train':
                self.root_path = root_path+"/RegDB/subset2/Thermal Images/train"
            elif mode=='validation':
                self.root_path = root_path+"/RegDB/subset2/Thermal Images/valid"
            else:
                self.root_path = root_path+"/RegDB/subset2/Thermal Images/test"
                
        
        self.paths = glob.glob(self.root_path+"/*.bmp")
        if mode =='train':
            random.shuffle(self.paths)
        self.transform = transform
        
        print("load image")
        _count = 0
        _count_max = len(self.paths)
        self.dict_img = {}
        for i_path in self.paths:
            _count += 1
            self.dict_img[i_path] = cv2.imread(i_path,0)
            print("\r", _count, " / ", _count_max, end="")
        
        
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        #_time = time.time()
        # img = cv2.imread(self.paths[idx],0)
        img = self.dict_img[self.paths[idx]]
        
        #print("Loading time", _time - time.time())
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        transformed = self.transform(image=img)
        image = transformed['image']
        label = gender_labeling(self.paths[idx])
        return image,label
    
    


        
class ConcatReg(RegDB):
    def __init__(self,datasets):
        
        
        self.content = dict()
        img_buf=[]
        label_buf=[]
        self.len_ds=[]
        self.transforms=[]
        for dataset in datasets:
            img_buf.extend(dataset.content['image'])
            label_buf.extend(dataset.content['label'])        
        self.content['image']=img_buf
        self.content['label']=label_buf
            
    def __len__(self):
        return len(self.content['image'])
    
    def __getitem__(self, idx):
        
        return self.content['image'][idx],self.content['label'][idx]
            


class Seg_Dataset(Dataset):
    def __init__(self, img_dir, transform):
        super().__init__()
        list_img = os.listdir(img_dir+"\\data")
        list_mask = os.listdir(img_dir+"\\mask")
        if "desktop.ini" in list_img:
            list_img.remove('desktop.ini')
        if "desktop.ini" in list_mask:
            list_mask.remove('desktop.ini')
        self.content={"image":[],"label":[]}
        self.list_img_path=sorted([os.path.join(img_dir,"data",i) for i in list_img])
        self.list_mask_path=sorted([os.path.join(img_dir,"mask",i) for i in list_mask])
        self.transform = transform
        for i,j in tqdm(zip(self.list_img_path,self.list_mask_path)):
            img = cv2.imread(i,0)
            mask = cv2.imread(j,0)
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)       
            transformed= self.transform(image = img, mask = mask)      
         
            self.content['image'].append()
            self.content['label'].append(mask)
    def __len__(self):
        return len(self.list_img_path) 
    
    def __getitem__(self, idx):
       
        return self.content['image'][idx],self.content['label'][idx]
    
class ConcatSeg(Seg_Dataset):
    
    def __init__(self,datasets):
        
        
        self.content = dict()
        img_buf=[]
        label_buf=[]
        self.len_ds=[]
        self.transforms=[]
        for dataset in datasets:
            img_buf.extend(dataset.content['image'])
            label_buf.extend(dataset.content['label'])        
        self.content['image']=img_buf
        self.content['label']=label_buf
            
    def __len__(self):
        return len(self.content['image'])
    
    def __getitem__(self, idx):
        
        return self.content['image'][idx],self.content['label'][idx]
            
            


if __name__ =='__main__':

    pass





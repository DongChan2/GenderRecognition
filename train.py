import torch
from torch.optim import *
from argparse import ArgumentParser
from dataset import RegDB
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import numpy as np 

from models.resnet_model import resnet101,resnet101_freezing,resnet34,resnet18
from mp_dataloader import DataLoader_multi_worker_FIX
import trainer

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    hparam ={'EPOCH':10,'BATCH_SIZE':5}
    
    
    root_path = "./Dataset"
    transform = {'origin': 
                            A.Compose([
                            A.Resize(384,128,always_apply=True),
                            A.ToFloat(max_value=255),
                            A.HorizontalFlip(p=0.5),
                            A.OneOf([A.Affine(translate_px={'x':(-30,30),'y':(-10,20)},keep_ratio=True,p=0.5),
                                    A.Affine(rotate=20,keep_ratio=True,p=0.5)],p=0.5),
                            A.RandomBrightnessContrast(brightness_limit=(-0.3,0.1),contrast_limit=(-0.2,0.3),p=0.3), 
                    
                            
                       
                            ToTensorV2()]),
                                          
               
                 
                  "valid": A.Compose([
                  
                            A.Resize(384,128),
                            A.ToFloat(max_value=255),
                    
                            ToTensorV2()
                          ]                   
                         )}
    train_ds = RegDB(root_path=root_path, transform=transform['origin'], K=1,mode='train')
    valid_ds = RegDB(root_path=root_path, transform=transform['valid'], K=1,mode='validation')
    test_ds = RegDB(root_path=root_path, transform=transform['valid'], K=2,mode='test')
    
    train_loader = DataLoader_multi_worker_FIX(dataset=train_ds,batch_size=hparam['BATCH_SIZE'],pin_memory=True, shuffle= True,num_workers=2)
    valid_loader = DataLoader_multi_worker_FIX(dataset=valid_ds,batch_size=1,pin_memory=True, shuffle= False,num_workers=2)
    test_loader = DataLoader_multi_worker_FIX(dataset=test_ds,batch_size=1,pin_memory=True, shuffle= False,num_workers=2)
    
 
    
    model = resnet101_freezing(pretrained = True)
    # model = resnet101(pretrained = False)
    # optimizer = AdamW(model.parameters(),lr=2e-6,capturable=True,weight_decay=2e-3)
    # optimizer = Adam(model.parameters(),lr=2e-6,weight_decay=1e-4,capturable=True)
    optimizer = Adam(model.parameters(),lr=2e-6,weight_decay=1e-4)#,capturable=True)
    # optimizer = SGD(model.parameters(),lr=0.0001,momentum=0.9,weight_decay=0.001)
    def lr_lambda(step,warmup_steps=10):
            if step < warmup_steps:
                return float(step) / float(max(1.0, warmup_steps))
            return 1.
    lr_scheduler= torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda x:x*0.9)
   
    
    # CKPT_PATH="D:/dongchan/Backup/data/GenderClassification/resnet101/2023_04_09 18_20_02/Epoch_0.pth.tar"
    trainer.train(model,train_loader,valid_loader,optimizer,hparam,device,lr_scheduler=lr_scheduler,save_ckpt=False)
    # trainer.resume_train(model,train_loader,valid_loader,optimizer,hparam,device,ckpt_path=CKPT_PATH)
    # trainer.test(model,test_loader,hparam,device,ckpt_path=CKPT_PATH)

if __name__ =='__main__':
    parser = ArgumentParser()
    parser.add_argument("--batch_size", default=16,type = int)
    args = parser.parse_args()
    main(args)

    
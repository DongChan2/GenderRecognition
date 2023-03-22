from dataloader import *
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset import *
from argparse import ArgumentParser
from torchvision.models import resnet101
from model.classification_model import *
from torch.utils.data import ConcatDataset
import torchvision



def main(args):
  
 #@@ dataset & dataloader
    seed_everything(42)    
    root_path = "C:\\Users\\dongchan\\VScodeProjects\\GenderRecognition\\Dataset\\PersonReID Data\\RegDB"  #dataset 기본 루트
    transform = {'origin':A.Compose([
                            A.Resize(32*12,32*4),
                  
                
                            A.Normalize(mean=(0.41373693,),std=(0.16828156,)),
                            ToTensorV2()
                          ]),
                 "aug_1":A.Compose([
                            A.Resize(32*12,32*4),
                            A.HorizontalFlip(p=1),
                            A.Normalize(mean=(0.41682844,),std=(0.1675136,)),
                            ToTensorV2()
                          ]),
                 "aug_2": A.Compose([
                            A.Resize(32*12,32*4),
                            A.Affine(translate_px={'x':(-20,20),'y':(-10,20)},keep_ratio=True,p=1),
                            A.Normalize(mean=(0.41682844,),std=(0.1675136,)),
                            ToTensorV2()
                          ]),
                 
                "valid": A.Compose([
                  
                            A.Resize(32*12,32*4),
                            A.Normalize(mean=(0.4099696,),std=(0.17402484,)),
                   
                            ToTensorV2()
                          ]                   
                         )}
    
    origin_ds = RegDB(root_path=root_path, transform=transform['origin'], K=2)
    aug1_ds =  RegDB(root_path=root_path, transform=transform['aug_1'], K=2)
    aug2_ds = RegDB(root_path=root_path, transform=transform['aug_2'], K=2)
    test_ds = RegDB(root_path=root_path, transform=transform['valid'], K=1)
    


    train_ds = ConcatReg([origin_ds,aug1_ds,aug2_ds])

    train_loader = RegDB_dl(train_ds,do_validation=False,sampling=args.sampling, batch_size=args.batch_size,num_workers = (args.num_workers))
    test_loader = RegDB_dl(test_ds,sampling=False,do_validation=False, shuffle=False, batch_size=16,num_workers = (args.num_workers))

    

        
    
    
    

#     # call back
#     dir_path = f"D:\\dongchan\\Backup\\data\\"+ args.model_path
#     os.makedirs(name=dir_path, exist_ok=True)
#     checkpoint_callback = [ModelCheckpoint(monitor='val_loss', save_top_k=5, every_n_epochs=1, filename='best_loss_{val_loss:.3f}',
#                                           dirpath=dir_path),
#                            ModelCheckpoint(monitor='val_acc', save_top_k=5, every_n_epochs=1, filename='best_acc_{val_acc:.2f}',mode='max',
#                                           dirpath=dir_path),
#                           ]

# #@@ model define
 
 
    

#     model = resnet_pl(name = 'resnet101',lr=args.lr)

#     logger = TensorBoardLogger(os.getcwd(), name=args.logger_path)

    
#     hparams = {"batch_size":args.batch_size, "lr":args.lr}
    
#     logger.log_hyperparams(hparams)
#     logger.save()
  
    
#     trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices, precision=16,
#                          callbacks=checkpoint_callback,
#                          max_epochs=20, benchmark=True, logger=logger,log_every_n_steps=10
#                          )

#     trainer.fit(model, train_loader, test_loader)
    # print(trainer.current_epoch)
  
 




if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument("--accelerator", default='gpu')
    parser.add_argument("--devices", default=1)
    parser.add_argument("--batch_size", default=16,type = int)
    parser.add_argument("--num_workers", default = 0 ,type = int)
    parser.add_argument("--lr",default=1e-5,type=float)
    # parser.add_argument("--logger_path",type=str)
    # parser.add_argument("--model_path",type=str)
    # parser.add_argument("--sampling",type=bool)
    

    args = parser.parse_args()

    main(args)




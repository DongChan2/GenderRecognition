from dataloader import *
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset import *
from argparse import ArgumentParser
from torchvision.models import resnet101
from model.classification_model import *
import matplotlib.pyplot as plt 

def visualize(data_loader):
    data_loader = iter(data_loader)
    fig,ax = plt.subplots(5,10)
    fig2,ax2 = plt.subplots(5,10)
    fig3,ax3 = plt.subplots(5,10)
    fig4,ax4 = plt.subplots(5,10)
    ax=ax.flatten()
    ax2 = ax2.flatten()
    ax3=ax3.flatten()
    ax4 = ax4.flatten()
    for i in range(200):
        x,y=next(data_loader)
    for i in range(200):
        x,y=next(data_loader)
        x=x.squeeze(0)
        x=(x.permute(1,2,0)).detach().cpu().numpy()
        y=y.squeeze(0)
        y=y.detach().cpu().numpy()
        
        if i<50:
            ax[i].imshow(x)
            ax[i].axis("off")
            ax[i].set_title(y[0])
        elif i<100:
            ax2[i-50].imshow(x)
            ax2[i-50].axis("off")
            ax2[i-50].set_title(y[0])    
        elif i<150:
            ax3[i-100].imshow(x)
            ax3[i-100].axis("off")
            ax3[i-100].set_title(y[0])    
        elif i<200:
            ax4[i-150].imshow(x)
            ax4[i-150].axis("off")
            ax4[i-150].set_title(y[0])    
    plt.show()
        
def main(args):
 #@@ dataset & dataloader
    root_path = "C:\\Users\\dongchan\\VScodeProjects\\GenderRecognition\\Dataset\\PersonReID Data\\RegDB"
    transform = A.Compose([
                            A.Resize(32 * 12, 32 * 4),
                            A.ToFloat(),
                            ToTensorV2()
                          ]
                         )

    test_ds = RegDB(root_path=root_path, transform=transform, K=2)
    # train_ds = RegDB(root_path=root_path, transform=transform, K=1)
    test_loader = RegDB_dl(test_ds,do_validation=False, batch_size=1,num_workers = (args.num_workers))
    # train_loader,valid_loader = RegDB_dl(train_ds,do_validation=True, batch_size=1,num_workers = (args.num_workers))


    # call back
    ckpt_path = "D:\\dongchan\\Backup\\data\\resnext_101\\fine_tuning\\fold-1\\checkpoints\\best_acc.ckpt"


#@@ model define

    model = resnet_pl(num_layers = 101,lr=args.lr)

   

    trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices, precision=16,benchmark=True,logger=False)
                         

    trainer.test(model,dataloaders=test_loader,ckpt_path =ckpt_path)
    # visualize(train_loader)
   



if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument("--accelerator", default='gpu')
    parser.add_argument("--devices", default=1)
    parser.add_argument("--batch_size", default=16)
    parser.add_argument("--num_workers", default = 0)
    parser.add_argument("--lr",default=1e-5)

    args = parser.parse_args()

    main(args)
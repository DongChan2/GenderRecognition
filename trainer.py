import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as TF
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

from tqdm import tqdm 
import time
from datetime import datetime
import os

from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.tensorboard import SummaryWriter
from mp_dataloader import DataLoader_multi_worker_FIX

def save_checkpoint(epoch,model,optimizer,loss,name,scheduler=None):
    PATH= "./checkpoints/"
    NEW_PATH=PATH+name
    os.makedirs(NEW_PATH,exist_ok=True)
    ckpt={"MODEL":model.state_dict(),
          "OPTIMIZER":optimizer.state_dict(),
          'EPOCH':epoch+1,
          "NAME":name}
    if scheduler is not None:
        ckpt.update({'SCHEDULER':scheduler,"SCHEDULER_STATE":scheduler.state_dict()})
    torch.save(ckpt,NEW_PATH+f"\\Epoch_{epoch+1}.pth.tar")
    
    


def compute_eer(preds,targets):
    

    fpr, tpr, thresholds = roc_curve(targets, preds, pos_label=1,drop_intermediate=True)

    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    print(eer)
 
    return eer

def compute_accuracy(preds,target):
    ##preds.shape = N,C
    ##target.shape= N
    preds = torch.softmax(preds,dim=1)
    preds = (torch.argmax(preds,dim=1)).view(-1)
    #target=target.view(-1)
    target=target.view(-1)
    accuracy = ((preds==target).double().sum())/len(target)
    return accuracy
    
class _Loss(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.loss_ce = nn.CrossEntropyLoss()
        
    def forward(self, pred, target):
        # pred B, 2
        # target B
        
   
        return self.loss_ce(pred, target)

def _train_one_step(model,data,optimizer,device,**kwargs):
    logger = kwargs['logger']
    criterion = kwargs['criterion']

    x,y = data
    x.requires_grad_(True)
    x=x.to(device)
    y=y.to(device)
    # print(x.requires_grad)
    # print(x.requires_grad)
    # loss = F.cross_entropy(model(x),y.long()) 
    #loss = criterion(model(x),y.long())
    #accuracy = compute_accuracy(model(x),y.long())
    pred = model(x)
    optimizer.zero_grad()
    loss = criterion(pred,y.long())
    accuracy = compute_accuracy(pred,y.long())
    
    logger.add_scalar("loss/step",loss,kwargs['iter'])
    logger.add_scalar("accuracy/step",accuracy,kwargs['iter'])
    
    loss.backward()
    optimizer.step()
    
    return {'loss':loss.item(),'accuracy':accuracy.item()}
    

def _train_one_epoch(model,dataloader,optimizer,device,**kwargs):
    model.train()
    total_loss = 0
    total_accuracy = 0
    
    for batch_index,data in enumerate(tqdm(dataloader)):
        
        history = _train_one_step(model,data,optimizer,device,logger=kwargs['logger'],iter=(len(dataloader)*(kwargs['epoch_index'])+(batch_index+1)), criterion = kwargs['criterion'])
        total_loss += history['loss']
        total_accuracy += history['accuracy']

    return {'loss':total_loss,'accuracy':total_accuracy}

def _validate_one_step(model,data,device,*args,**kwargs):
    logger = kwargs['logger']
    criterion = kwargs['criterion']
    x,y = data
    x=x.to(device)
    y=y.to(device)
    # loss = F.cross_entropy(model(x),y.long())
    #loss = criterion(model(x),y.long())
    #accuracy = compute_accuracy(model(x),y.long())
    pred = model(x)
    loss = criterion(pred,y.long())
    accuracy = compute_accuracy(pred,y.long())
   
    logger.add_scalar("loss/step",loss,kwargs['iter'])
    logger.add_scalar("accuracy/step",accuracy,kwargs['iter'])
    
    return {'loss':loss.item(),'accuracy':accuracy.item()}
    

def _validate_one_epoch(model,dataloader,device,**kwargs):
    model.eval()
    total_loss = 0
    total_accuracy = 0
    for batch_index,data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            history = _validate_one_step(model,data,device,logger=kwargs['logger'],iter=(len(dataloader)*(kwargs['epoch_index'])+(batch_index+1)), criterion = kwargs['criterion'])
        total_loss += history['loss']
        total_accuracy += history['accuracy']
        
    
    return {'loss':total_loss,'accuracy':total_accuracy}




def train(model,dataloader,valid_dataloader,optimizer,hparam,device,lr_scheduler=None,save_ckpt=True):
    print("Training Start")
    print("="*20)
    t= datetime.today().strftime("%Y_%m_%d %H_%M_%S")
    #t="layer2,layer3,layer4,fc"
    train_logger = SummaryWriter(log_dir = f"./logs/{t}/train")
    valid_logger = SummaryWriter(log_dir = f"./logs/{t}/validation")

    model.to(device)
    epochs = hparam['EPOCH']
    
    criterion = nn.CrossEntropyLoss()
    
    for idx,epoch in (enumerate(range(epochs))):
        
        print(f"\rEpoch :{idx+1}/{epochs}")
        history = _train_one_epoch(model,dataloader,optimizer,device,epoch_index=idx,logger=train_logger,criterion=criterion)
        epoch_loss = history['loss'] / len(dataloader)
        epoch_accuracy = history['accuracy'] / len(dataloader)
        
        train_logger.add_scalar("loss/epoch",epoch_loss,idx+1)
        train_logger.add_scalar("accuracy/epoch",epoch_accuracy,idx+1)
        val_history = _validate_one_epoch(model,valid_dataloader,device,epoch_index=idx,logger=valid_logger,criterion=criterion)
        epoch_val_loss = val_history['loss'] / len(valid_dataloader)
        epoch_val_accuracy = val_history['accuracy'] / len(valid_dataloader)
        valid_logger.add_scalar("loss/epoch",epoch_val_loss,idx+1)
        valid_logger.add_scalar("accuracy/epoch",epoch_val_accuracy,idx+1)
        if lr_scheduler is not None:
            lr_scheduler.step()
        if save_ckpt:
            save_checkpoint(epoch,model,optimizer,epoch_loss,name=t)
    train_logger.close()
    valid_logger.close()
    print('Training End')
    print("="*20)

def _test_one_step(model,data,device,*args,**kwargs):
    
    x,y=data
    x=x.to(device)
    y=y.to(device)

    pred = model(x)
    return pred,y
    

def _test_one_epoch(model,dataloader,device,*args,**kwargs):
    model.eval()
    model.to(device)
    criterion=kwargs['criterion']
    total_loss =0
    total_acc=0
    total_pred=torch.tensor([],device=device)
    total_y=torch.tensor([],device=device)


    for data in tqdm(dataloader):
        with torch.no_grad():
            pred,y = _test_one_step(model,data,device)

        total_loss+=criterion(pred,y)
        total_acc+=compute_accuracy(pred,y)
        total_pred=torch.cat([total_pred,torch.softmax(pred,dim=1)[:,1].view(-1)],dim=0)
        total_y=torch.cat([total_y,y.view(-1)],dim=0)
    eer = compute_eer(total_pred.detach().cpu(),total_y.detach().cpu())
    acc = total_acc/len(dataloader)
    loss = total_loss/len(dataloader)

    result = {'eer':eer,'loss':loss,'acc':acc}
    return result 

def test(model,dataloader,hparam,device,ckpt_path):
    print("Testing Start")
    print("="*20)
    ckpt = torch.load(ckpt_path)
    ckpt_model = ckpt['MODEL']
    model.load_state_dict(ckpt_model)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    test_history = _test_one_epoch(model,dataloader,device,criterion=criterion)
    print(test_history)
    print("Testing End")
    print("="*20)
    
def resume_train(model,dataloader,valid_dataloader,optimizer,hparam,device,ckpt_path:str,lr_scheduler=None,save_ckpt=True):
    print("Resume Training")
    print("="*20)
    ckpt = torch.load(ckpt_path)
    name = ckpt['NAME']
    model.load_state_dict(ckpt['MODEL'])
    
    optimizer.load_state_dict(ckpt['OPTIMIZER'])
    if 'SCHEDULER' in ckpt.keys():
        lr_scheduler=ckpt['SCHEDULER']
        lr_scheduler.load_state_dict(ckpt['SCHEDULER_STATE'])
    if device == 'cuda':
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
        for state in lr_scheduler.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
    
    start_epoch = ckpt['EPOCH']
    end_epoch = hparam['EPOCH']
    
    
    train_logger = SummaryWriter(log_dir = f"./logs/{name}/train")
    valid_logger = SummaryWriter(log_dir = f"./logs/{name}/validation")

    model=model.to(device)
   
   
    
    criterion = nn.CrossEntropyLoss()
    
    for epoch in ((range(start_epoch,end_epoch))):
        
        print(f"\rEpoch :{epoch+1}/{end_epoch}")
        print("="*100)
        #@@ Training Phase
        history = _train_one_epoch(model,dataloader,optimizer,device,epoch_index=epoch,logger=train_logger,criterion=criterion)
        epoch_loss = history['loss'] / len(dataloader)
        epoch_accuracy = history['accuracy'] / len(dataloader)
        train_logger.add_scalar("loss/epoch",epoch_loss,epoch+1)
        train_logger.add_scalar("accuracy/epoch",epoch_accuracy,epoch+1)
        
        #@@ Validation Phase
        val_history = _validate_one_epoch(model,valid_dataloader,device,epoch_index=epoch,logger=valid_logger,criterion=criterion)
        epoch_val_loss = val_history['loss'] / len(valid_dataloader)
        epoch_val_accuracy = val_history['accuracy'] / len(valid_dataloader)
        valid_logger.add_scalar("loss/epoch",epoch_val_loss,epoch+1)
        valid_logger.add_scalar("accuracy/epoch",epoch_val_accuracy,epoch+1)
        if lr_scheduler is not None:
            lr_scheduler.step()
        if save_ckpt:
            save_checkpoint(epoch,model,optimizer,epoch_loss,name=name)
    train_logger.close()
    valid_logger.close()
    print("Resume End")
    print("="*20)



if __name__ == '__main__':
    pass
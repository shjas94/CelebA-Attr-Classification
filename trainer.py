import os
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataset.dataset import CelebA
from dataset.augmentations import get_train_transforms, get_valid_transforms
from utils.utils import *
from utils.scheduler import CosineAnnealingWarmUpRestart
from utils.earlystopping import EarlyStopping
import wandb

def train_one_epoch(epoch, 
                    device, 
                    model, 
                    loader, 
                    optimizer, 
                    criterion):
    model.train()
    smiling_acc_list, wavy_acc_list, male_acc_list, train_loss_list = list(), list(), list(), list()
    pbar = tqdm(loader, total=loader.__len__(), position=0, leave=True)
    for sample in pbar:
        optimizer.zero_grad()
        imgs, labels = sample['imgs'].float().to(device), sample['labels'].float().to(device)
        probs = model(imgs)
        loss = criterion(probs, labels)
        loss.backward()
        optimizer.step()
        
        probs, labels = probs.cpu().detach().numpy(), labels.cpu().detach().numpy()
        learning_rate = get_lr(optimizer)
        
        train_loss = loss.item()
        train_loss_list.append(train_loss)
        
        train_accs = calculate_acc(probs, labels) # (smiling_acc, wavy_acc, male_acc)
        smiling_acc_list.append(train_accs[0])
        wavy_acc_list.append(train_accs[1])
        male_acc_list.append(train_accs[2])
        
        desc = f"Train Epoch : {epoch+1}, Loss : {np.mean(train_loss_list):.2f}, Smiling Accuracy : {np.mean(smiling_acc_list):.2f}, Wavy Hair Accuracy : {np.mean(wavy_acc_list):.2f}, Male Accuracy : {np.mean(male_acc_list):.2f}"
        pbar.set_description(desc)
    return learning_rate, np.mean(train_loss_list), np.mean(smiling_acc_list), np.mean(wavy_acc_list), np.mean(male_acc_list)



def valid_one_epoch(epoch, 
                    device, 
                    model, 
                    loader, 
                    criterion):
    model.eval()
    smiling_acc_list, wavy_acc_list, male_acc_list, valid_loss_list = list(), list(), list(), list()
    pbar = tqdm(loader, total=loader.__len__(), position=0, leave=True)
    for sample in pbar:
        imgs, labels = sample['imgs'].float().to(device), sample['labels'].float().to(device)
        probs = model(imgs)
        loss = criterion(probs, labels)
        
        probs, labels = probs.cpu().detach().numpy(), labels.cpu().detach().numpy()
        
        valid_loss = loss.item()
        valid_loss_list.append(valid_loss)
        
        valid_accs = calculate_acc(probs, labels) # (smiling_acc, wavy_acc, male_acc)
        smiling_acc_list.append(valid_accs[0])
        wavy_acc_list.append(valid_accs[1])
        male_acc_list.append(valid_accs[2])
             
        desc = f"Valid Epoch : {epoch+1}, Loss : {np.mean(valid_loss_list):.2f}, Smiling Accuracy : {np.mean(smiling_acc_list):.2f}, Wavy Hair Accuracy : {np.mean(wavy_acc_list):.2f}, Male Accuracy : {np.mean(male_acc_list):.2f}"
        pbar.set_description(desc)
    return np.mean(valid_loss_list), np.mean(smiling_acc_list), np.mean(wavy_acc_list), np.mean(male_acc_list)



def trainer(cfg, model):
    seed_everything(cfg.seed_num)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    if use_cuda:
        torch.cuda.empty_cache()
        
    train_partition = get_partition(cfg, 0)
    valid_partition = get_partition(cfg, 1)
    
    train_sampler = SubsetRandomSampler(train_partition)
    valid_sampler = SubsetRandomSampler(valid_partition)
    
    train_transforms = get_train_transforms(cfg)
    valid_transforms = get_valid_transforms(cfg)
    
    train_dataset = CelebA(cfg, transforms=train_transforms)
    valid_dataset = CelebA(cfg, transforms=valid_transforms)
    
    train_loader = DataLoader(train_dataset, cfg.train_batch_size, drop_last=True, sampler=train_sampler)
    valid_loader = DataLoader(valid_dataset, cfg.valid_batch_size, drop_last=False, sampler=valid_sampler)
    
    if cfg.wandb:
        wandb.init(project='CelebA',
                   group=cfg.model,
                   name=cfg.run_name, config=cfg)
               
        wandb.watch(model)
        
    optimizer = get_optimizer(cfg, model)
    criterion = torch.nn.BCELoss()
    best_valid_loss = np.inf
    best_model = None
    early_stopping = EarlyStopping(patience=cfg.patience, 
                                   verbose=True, 
                                   path=os.path.join(cfg.save_path, f'checkpoint.pt'))
    
    if cfg.scheduler == 'none':
        scheduler = None
    elif cfg.scheduler == 'cosinewarmup':
        scheduler =  CosineAnnealingWarmUpRestart(optimizer=optimizer, 
                                                  T_0=4, 
                                                  T_mult=1, 
                                                  eta_max=2e-4,  
                                                  T_up=1, 
                                                  gamma=0.5)

    save_path = os.path.join(cfg.save_path, cfg.run_name)
    model.to(device)
    for e in range(cfg.epoch):
        with torch.set_grad_enabled(True):
            lr, train_loss, train_smiling_acc, train_wavy_acc, train_male_acc = \
                train_one_epoch(epoch=e, 
                                device=device, 
                                model=model, 
                                loader=train_loader, 
                                optimizer=optimizer, 
                                criterion=criterion) # lr : for wandb logging
            if cfg.wandb:
                wandb.log({
                    "Train Loss": train_loss,
                    "Train Smiling Accuracy": train_smiling_acc,
                    "Train Wavy Hair Accuracy": train_wavy_acc,
                    "Train Male Accuracy": train_male_acc,
                    "Learning Rate": lr
                })
        with torch.no_grad():
            valid_loss, valid_smiling_acc, valid_wavy_acc, valid_male_acc = \
                valid_one_epoch(epoch=e, 
                                device=device, 
                                model=model, 
                                loader=valid_loader, 
                                criterion=criterion)
            if cfg.wandb:
                wandb.log({
                    "Valid Loss": valid_loss,
                    "Valid Smiling Accuracy": valid_smiling_acc,
                    "Valid Wavy Hair Accuracy": valid_wavy_acc,
                    "Valid Male Accuracy": valid_male_acc
                })
                
        if best_valid_loss > valid_loss:
            best_valid_loss = valid_loss
            best_model = model
            best_epoch = e
        if scheduler:
            scheduler.step()
            
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print('Apply Early Stopping.....')
            torch.save(best_model, f'{save_path}_{e}_{best_valid_loss:.2f}.pth')
            if cfg.wandb:
                wandb.join()
            break
        
    torch.save(best_model, f'{save_path}_{best_epoch}_{best_valid_loss:.2f}.pth')
    if cfg.wandb:
        wandb.join()
    return best_model
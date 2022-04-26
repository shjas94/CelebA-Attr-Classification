import os
import random
import numpy as np
import pandas as pd
import torch
from zipfile import ZipFile
from models.models import CustomResNet, CustomEfficientNet, effnetv2_s, effnetv2_m

def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    
def make_dirs(dirs):
    for dir in dirs:
        if not os.path.isdir(dir):
            os.mkdir(dir)
            print("Done!!")
        else:
            print("Dir already exists!!")
    
def unzip_img_files(path):
    imgs_zip_file = os.path.join(path, 'img_align_celeba.zip')
    img_path = os.path.join(path, 'img_align_celeba')
    if os.path.exists(imgs_zip_file):
        with ZipFile(imgs_zip_file, 'r') as zip:
            if not os.path.isdir(img_path):
                print('Extracting Image Files..... Just wait for a moment!!')
                zip.extractall(path)
                print('Finished')
            else:
                print('Image Files already extracted')
    else:
        raise FileNotFoundError
            
def unzip_files(path, source_file, target_file):
    source_path = os.path.join(path, source_file)
    target_path = os.path.join(path, target_file)
    if os.path.exists(path):
        with ZipFile(source_path, 'r') as zip:
                if not os.path.isfile(target_path):
                    print(f'Extracting {source_file}..... Just wait for a moment!!')
                    zip.extractall(path)
                    print('Finished')
                else:
                    print(f'{target_file} already extracted')
    else:
        raise FileNotFoundError
    

def get_partition(cfg, partition):
    '''
                 0 : Train
    partition == 1 : Validation
                 2 : Test
    '''
    partition_info = pd.read_csv(os.path.join(cfg.data_path,'list_eval_partition.csv'))
    return list(partition_info[partition_info['partition'] == partition].index)

def get_optimizer(cfg, model):
    if cfg.optimizer == 'adamw':
        return torch.optim.AdamW(params=model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'adam':
        return torch.optim.Adam(params=model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise ValueError
    
def get_model(model_name):
    if model_name == 'customresnet':
        return CustomResNet()
    elif model_name == 'customefficientnet':
        return CustomEfficientNet()
    elif model_name == 'effnetv2s':
        return effnetv2_s()
    else:
        raise ValueError
    
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def calculate_acc(probs, labels):
    preds_Smiling = probs[:,0] > 0.5
    preds_Wavy_Hair = probs[:,1] > 0.5
    preds_Male = probs[:,2] > 0.5    
    Smiling_acc = (labels[:,0] == preds_Smiling).mean()
    Wavy_Hair_acc = (labels[:,1] == preds_Wavy_Hair).mean()
    Male_acc = (labels[:,2] == preds_Male).mean()
    return (Smiling_acc, Wavy_Hair_acc, Male_acc)

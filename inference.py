import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataset.augmentations import get_transforms
from tqdm import tqdm
from dataset.dataset import CelebA, CustomDataset
from utils.utils import calculate_acc, get_partition, \
    unzip_img_files, unzip_files
    
@torch.no_grad()
def inference_celeba(model, 
                     device, 
                     loader, 
                     criterion):
    
    smiling_acc_list = list()
    wavy_acc_list = list()
    male_acc_list = list()
    test_loss_list = list()

    model.eval()
    pbar = tqdm(loader, total=loader.__len__(), position=0, leave=True)
    for sample in pbar:
        imgs, labels = sample['imgs'].float().to(device), sample['labels'].float().to(device)
        probs = model(imgs)
        loss = criterion(probs, labels)
        probs, labels = probs.cpu().detach().numpy(), labels.cpu().detach().numpy()
            
        test_loss = loss.item()
        test_loss_list.append(test_loss)
            
        test_accs = calculate_acc(probs, labels) # (smiling_acc, wavy_acc, male_acc)
        smiling_acc_list.append(test_accs[0])
        wavy_acc_list.append(test_accs[1])
        male_acc_list.append(test_accs[2])
        
        desc = f"Loss : {np.mean(test_loss_list):.2f}, " \
            f"Smiling Accuracy : {np.mean(smiling_acc_list):.2f}, " \
            f"Wavy Hair Accuracy : {np.mean(wavy_acc_list):.2f}, " \
            f"Male Accuracy : {np.mean(male_acc_list):.2f}"
        pbar.set_description(desc)
        
    return (np.mean(smiling_acc_list), np.mean(wavy_acc_list), np.mean(male_acc_list))


@torch.no_grad()
def inference_custom(model, 
                     device, 
                     loader):
    probs_list = []
        
    model.eval()
    pbar = tqdm(loader, total=loader.__len__(), position=0, leave=True)
    for sample in pbar:
        imgs = sample['imgs'].float().to(device)
        probs = model(imgs)
        probs = probs.cpu().detach().numpy()
        probs_list.append(probs)    
    probs_arr = probs_list[0]
    print(probs_list[0].shape)
    for i in range(1, len(probs_list)):
        probs_arr = np.concatenate([probs_arr, probs_list[i]], axis=0)
    return probs_arr

def inference_main(cfg):
    if cfg.inference_type == 'celeba':
        
        ############## Unzip Files ##############
        
        unzip_img_files(cfg.data_path)
        unzip_files(cfg.data_path, 'list_eval_partition.zip', 'list_eval_partition.csv')
        unzip_files(cfg.data_path, 'list_attr_celeba.zip', 'list_attr_celeba.csv')
        
        #########################################
        
        ############ Prepare for data ############
        
        test_partition = get_partition(cfg.data_path, 'test')
        test_sampler = SubsetRandomSampler(test_partition)
        test_transforms = get_transforms(cfg.img_size, mode='test')
        test_dataset = CelebA(cfg.data_path, transforms=test_transforms)
        test_loader = DataLoader(test_dataset, cfg.test_batch_size, drop_last=False, sampler=test_sampler)
        
        ##########################################
        
        ############ Prepare for Model ############
        
        model_path = os.path.join(cfg.ckpt_path, cfg.model_name)
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        torch.cuda.empty_cache()
        model = torch.load(model_path).to(device)
        criterion = torch.nn.BCELoss()
        
        ###########################################
        
        accuracies = inference_celeba(model, device, test_loader, criterion)
        print(f"\n-----CelebA test set Inference Result-----\n" \
            f"Smiling Acc : {accuracies[0]:.2f}, " \
            f"Wavy Hair Acc : {accuracies[1]:.2f}, " \
            f"Male Acc : {accuracies[2]:.2f}")
                
    else:
        '''
        Just for personal experiment
        '''
        ############ Prepare for data ############
        
        test_transforms = get_transforms(cfg.img_size, mode='test')
        test_dataset = CustomDataset(cfg.data_path, transforms=test_transforms)
        test_loader = DataLoader(test_dataset, cfg.test_batch_size, drop_last=False)
        
        ##########################################
        
        ############ Prepare for Model ############
        
        model_path = os.path.join(cfg.ckpt_path, cfg.model_name)
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        torch.cuda.empty_cache()
        model = torch.load(model_path).to(device)
        
        ###########################################
        
        ############ Calculate Answer ############
        probs = inference_custom(model, device, test_loader)
        experiment_name = cfg.data_path.split('/')[2]
        if experiment_name == 'negate_female_wavy_drop_first':
            '''
            Wavy Hair & Not Female
            Ignore smiling accuracies
            '''
            labels = np.tile([[1, 1, 1]], reps=[probs.shape[0], 1])
            accuracies = calculate_acc(probs, labels)
            print(f"\n-----Wavy Hair & Not Female Inference Result-----\n" \
            f"Wavy Hair Acc : {accuracies[1]:.2f}, " \
            f"Not Female Acc : {accuracies[2]:.2f}\n")
            
        elif experiment_name == 'negate_female_wavy_drop_second':
            '''
            Not Wavy Hair & FeMale
            Ignore smiling accuracies
            '''
            labels = np.tile([[1, 0, 0]], reps=[probs.shape[0], 1])
            accuracies = calculate_acc(probs, labels)
            print(f"\n-----Not Wavy Hair & Female Inference Result-----\n" \
            f"Not Wavy Hair Acc : {accuracies[1]:.2f}, " \
            f"Female Acc : {accuracies[2]:.2f}\n")
            
        elif experiment_name == 'negate_smiling_wavy_drop_first':
            '''
            Not Smiling & Wavy Hair
            Ignore Male accuracies
            '''
            labels = np.tile([[0, 1, 1]], reps=[probs.shape[0], 1])
            accuracies = calculate_acc(probs, labels)
            print(f"\n-----Not Smiling & Wavy Hair Inference Result-----\n" \
            f"Not Smiling Acc : {accuracies[0]:.2f}, " \
            f"Wavy Hair Acc : {accuracies[1]:.2f}\n")
            
        elif experiment_name == 'negate_smiling_wavy_drop_second':
            '''
            Smiling & Not Wavy Hair
            Ignore Male accuracies
            '''
            labels = np.tile([[1, 0, 1]], reps=[probs.shape[0], 1])
            accuracies = calculate_acc(probs, labels)
            print(f"\n-----Smiling & Not Wavy Hair Inference Result-----\n" \
            f"Smiling Acc : {accuracies[0]:.2f}, " \
            f"Not Wavy Hair Acc : {accuracies[1]:.2f}\n")
        ##########################################
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data', help='Root path of data')
    parser.add_argument('--ckpt_path', type=str, default='./checkpoint', help='Path of model checkpoint')
    parser.add_argument('--img_size', type=int, default=224, help='Size of input image')
    parser.add_argument('--test_batch_size', type=int, default=4, help='Batch size for inference')
    parser.add_argument('--model_name', type=str, help='Name of model checkpoint file')
    parser.add_argument('--inference_type', type=str, default='celeba', choices=('celeba', 'custom'))
    cfg = parser.parse_args()
    inference_main(cfg)
    
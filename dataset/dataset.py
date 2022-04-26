import os
from glob import glob
from zipfile import ZipFile
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class CelebA(Dataset):
    def __init__(self, cfg, transforms=None):
        super().__init__()
        self.root_path = cfg.data_path # root path of dataset
        self.img_path = os.path.join(self.root_path, 'img_align_celeba')
        self.transforms = transforms
        self.preprocess_img_id()
        self.preprocess_label()

    def preprocess_img_id(self):
        img_ids = list(glob(os.path.join(self.img_path, '*.jpg')))
        self.img_ids_sorted = sorted(img_ids, key=lambda x:int(x.split('/')[-1].split('.')[0]))

    def preprocess_label(self):
        labels = pd.read_csv(os.path.join(self.root_path, 'list_attr_celeba.csv'))
        labels = labels[['Smiling','Wavy_Hair', 'Male']]
        labels = labels.replace(-1, 0)
        self.labels = labels.loc[:,:].values

    def __getitem__(self, index):
        imgs = cv2.imread(self.img_ids_sorted[index], cv2.IMREAD_COLOR)
        imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)/255.
        if self.transforms:
            imgs = self.transforms(imgs)
        labels = torch.tensor(self.labels[index, :])
        sample = {'imgs':imgs, 'labels':labels}
        return sample
    
    def __len__(self):
        return len(self.img_ids_sorted)
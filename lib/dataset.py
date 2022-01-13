import json

import cv2
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule


torch.multiprocessing.set_sharing_strategy('file_system')


class CustomDataset(Dataset):
    def __init__(
        self, 
        files, 
        csv_feature_dict, 
        label_encoder,
        labels=None,
        transforms=None,
        mode='train',
    ):
        self.mode = mode
        self.files = files
        
        self.csv_feature_dict = csv_feature_dict
        
        if files is not None:
            self.csv_feature_check = [0]*len(self.files)
            self.csv_features = [None]*len(self.files)
            
        self.max_len = 24 * 6
        
        self.label_encoder = label_encoder
        
        self.transforms = transforms

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, i):
        file = self.files[i]
        file_name = file.split('/')[-1]
        
        # csv
        if self.csv_feature_check[i] == 0:
            csv_path = f'{file}/{file_name}.csv'
            df = pd.read_csv(csv_path)[self.csv_feature_dict.keys()]
            df = df.replace('-', 0)
            # MinMax scaling
            for col in df.columns:
                df[col] = df[col].astype(float) - self.csv_feature_dict[col][0]
                df[col] = df[col] / (self.csv_feature_dict[col][1]-self.csv_feature_dict[col][0])
            # zero padding
            pad = np.zeros((self.max_len, len(df.columns)))
            length = min(self.max_len, len(df))
            pad[-length:] = df.to_numpy()[-length:]
            # transpose to sequential data
            csv_feature = pad.T
            self.csv_features[i] = csv_feature
            self.csv_feature_check[i] = 1
        else:
            csv_feature = self.csv_features[i]
        
        # image
        image_path = f'{file}/{file_name}.jpg'
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transforms is not None:
            img = self.transforms(image=img)['image']
        
        if self.mode == 'train':
            json_path = f'{file}/{file_name}.json'
            with open(json_path, 'r') as f:
                json_file = json.load(f)
            
            crop = json_file['annotations']['crop']
            disease = json_file['annotations']['disease']
            risk = json_file['annotations']['risk']
            label = f'{crop}_{disease}_{risk}'
            
            return {
                'img': img,
                'csv_feature': torch.tensor(csv_feature, dtype=torch.float32),
                'label': torch.tensor(self.label_encoder[label], dtype=torch.long)
            }
        else:
            return {
                'img': img,
                'csv_feature': torch.tensor(csv_feature, dtype=torch.float32)
            }


class CustomDataModule(LightningDataModule):
    def __init__(
        self,
        train=None,
        val=None,
        test=None,
        csv_feature_dict=None,
        label_encoder=None,
        train_transforms=None,
        val_transforms=None,
        predict_transforms=None,
        num_workers=32,
        batch_size=8,
    ):
        super().__init__()
        self.train = train
        self.val = val
        self.test = test
        self.csv_feature_dict = csv_feature_dict
        self.label_encoder = label_encoder
        assert self.csv_feature_dict is not None
        assert self.label_encoder is not None
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.predict_transforms = predict_transforms
        self.num_workers = num_workers
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = CustomDataset(
            self.train, 
            self.csv_feature_dict,
            self.label_encoder,
            transfroms=self.train_transforms,
        )
        self.valid_dataset = CustomDataset(
            self.val, 
            self.csv_feature_dict,
            self.label_encoder,
            transforms=self.train_transforms,
        )
        self.predict_dataset = CustomDataset(
            self.test, 
            self.csv_feature_dict,
            self.label_encoder,
            transforms=self.predict_transforms,
            mode='test'
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        
    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

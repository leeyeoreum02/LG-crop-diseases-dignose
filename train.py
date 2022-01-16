import os
import argparse
from glob import glob

import pandas as pd
import numpy as np

from pytorch_lightning.plugins import DDPPlugin
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything
import albumentations as A
from albumentations.pytorch import ToTensorV2

from lib.model_effnet import Effnetb02LSTMModel, Effnetb32LSTMModel
from lib.model_effnet import Effnetb72LSTMModel, Effnetb7NS2LSTMModel
from lib.model_effnet import Effnetb7NS, Effnetb7NSPlus2LSTMModel
from lib.dataset import CustomDataModule, CustomDataModuleV2
from lib.utils import split_data, initialize, split_kfold


def get_args():
    parser = argparse.ArgumentParser(description='Training Effnet')
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    args = parser.parse_args()
    return args


def get_train_transforms(height, width):
    return A.Compose([
        A.Resize(height=height, width=width),
        A.CLAHE(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ColorJitter(p=0.2),
        A.RGBShift(p=0.2),
        A.RandomSnow(p=0.2),
        A.RandomResizedCrop(height=height, width=width, p=0.4),
        A.ShiftScaleRotate(
            scale_limit=0.2, 
            rotate_limit=10, 
            p=0.4
        ),
        A.HorizontalFlip(p=0.2),
        A.VerticalFlip(p=0.2),
        A.Rotate(p=0.2),
        A.RandomRotate90(p=0.2),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_valid_transforms(height, width):
    return A.Compose([
        A.Resize(height=height, width=width),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def train(model_name, args, csv_feature_dict, label_encoder, seed=42):
    """
    Use for model trained image and time series.
    """
    train_data, val_data = split_data(seed=seed, mode='train')
    
    train_transforms = get_train_transforms(args.height, args.width)
    val_transforms = get_valid_transforms(args.height, args.width)
    
    data_module = CustomDataModule(
        train=train_data,
        val=val_data,
        csv_feature_dict=csv_feature_dict,
        label_encoder=label_encoder,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )
    
    # model = Effnetb02LSTMModel(
    #     max_len=24*6, 
    #     embedding_dim=512, 
    #     num_features=len(csv_feature_dict), 
    #     class_n=len(label_encoder),
    #     rate=args.dropout_rate,
    #     learning_rate=args.lr,
    # )
    
    # model = Effnetb32LSTMModel(
    #     max_len=24*6, 
    #     embedding_dim=512, 
    #     num_features=len(csv_feature_dict), 
    #     class_n=len(label_encoder),
    #     rate=args.dropout_rate,
    #     learning_rate=args.lr,
    # )

    # model = Effnetb72LSTMModel(
    #     max_len=24*6, 
    #     embedding_dim=512, 
    #     num_features=len(csv_feature_dict), 
    #     class_n=len(label_encoder), 
    #     rate=args.dropout_rate,
    #     learning_rate=args.lr,
    # )
    
    model = Effnetb7NS2LSTMModel(
        max_len=24*6, 
        embedding_dim=512, 
        num_features=len(csv_feature_dict), 
        class_n=len(label_encoder), 
        rate=args.dropout_rate,
        learning_rate=args.lr,
    )
    
    ckpt_path = f'./weights/{model_name}/'
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_score',
        dirpath=ckpt_path,
        filename='{epoch}-{val_score:.2f}',
        save_top_k=-1,
        mode='max',
        save_weights_only=True,
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        gpus=[0, 1, 2, 3],
        # gpus=1,
        strategy=DDPPlugin(find_unused_parameters=False),
        # strategy=DDPPlugin(find_unused_parameters=True),
        precision=16,
        callbacks=[checkpoint_callback],
        log_every_n_steps=5,
    )

    trainer.fit(model, data_module)
    

def train_fold(model_name, fold, args):
    """
    Use for model trained only image.
    """
    df = pd.read_csv('./data/kfold.csv')
    df_train = df[df['kfold'] != fold].reset_index(drop=True)
    df_valid = df[df['kfold'] == fold].reset_index(drop=True)
    train_fold_id = list(df_train['id'])
    valid_fold_id = list(df_valid['id'])
    
    train_transforms = get_train_transforms(args.height, args.width)
    val_transforms = get_valid_transforms(args.height, args.width)
    
    data_module = CustomDataModuleV2(
        train_idx=train_fold_id,
        val_idx=valid_fold_id,
        root_path='data',
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )
    
    model = Effnetb7NS()
    
    ckpt_path = f'./weights/{model_name}/'
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_score',
        dirpath=ckpt_path,
        filename='{epoch}-{val_score:.2f}',
        save_top_k=-1,
        mode='max',
        save_weights_only=True,
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        gpus=[0, 1, 2, 3],
        # gpus=1,
        strategy=DDPPlugin(find_unused_parameters=False),
        # strategy=DDPPlugin(find_unused_parameters=True),
        precision=16,
        callbacks=[checkpoint_callback],
        log_every_n_steps=5,
    )

    trainer.fit(model, data_module)
    
    
def train_fold_v2(model_name, fold, args, csv_feature_dict, label_encoder):
    """
    Use for model trained image and time series.
    """
    df = pd.read_csv('./data/kfold.csv')
    df_train = df[df['kfold'] != fold].reset_index(drop=True)
    df_valid = df[df['kfold'] == fold].reset_index(drop=True)
    train_fold_id = df_train['id']
    valid_fold_id = df_valid['id']
    
    data = np.array(sorted(glob('data/train/*')))
    train = data[train_fold_id]
    val = data[valid_fold_id]
        
    train_transforms = get_train_transforms(args.height, args.width)
    val_transforms = get_valid_transforms(args.height, args.width)
    
    data_module = CustomDataModule(
        train=train,
        val=val,
        csv_feature_dict=csv_feature_dict,
        label_encoder=label_encoder,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )
    
    # model = Effnetb7NS2LSTMModel(
    #     max_len=24*6, 
    #     embedding_dim=512, 
    #     num_features=len(csv_feature_dict), 
    #     class_n=len(label_encoder), 
    #     rate=args.dropout_rate,
    #     learning_rate=args.lr,
    # )
    
    model = Effnetb7NSPlus2LSTMModel(
        max_len=24*6, 
        embedding_dim=512, 
        num_features=len(csv_feature_dict), 
        class_n=len(label_encoder), 
        rate=args.dropout_rate,
        learning_rate=args.lr,
        max_epochs=args.max_epochs,
    )
    
    ckpt_path = f'./weights/{model_name}/'
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_score',
        dirpath=ckpt_path,
        filename='{epoch}-{val_score:.2f}',
        save_top_k=-1,
        mode='max',
        save_weights_only=True,
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        gpus=[2, 3],
        # gpus=1,
        strategy=DDPPlugin(find_unused_parameters=False),
        # strategy=DDPPlugin(find_unused_parameters=True),
        precision=16,
        callbacks=[checkpoint_callback],
        log_every_n_steps=5,
    )

    trainer.fit(model, data_module)


def main():
    seed = 42
    seed_everything(seed)
    
    csv_feature_dict, label_encoder, _ = initialize()
    k = 5
    split_kfold(k=k, seed=seed)
    
    args = get_args()
    
    # train('effnetb7ns-lstm-512', args, csv_feature_dict, label_encoder, seed)
    # for fold in range(k):
    #     train_fold(f'effnetb7ns-w{args.width}-h{args.height}-f{fold}', fold, args)
    for fold in range(k):
        train_fold_v2(
            f'effnetb7nsplus-lstm-w{args.width}-h{args.height}-f{fold}-aug-sch', 
            fold, args, csv_feature_dict, label_encoder
        )


if __name__ == '__main__':
    main()

import os
import argparse
from glob import glob
from typing import Callable, Sequence, Dict, List

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

from pytorch_lightning.plugins import DDPPlugin
import pytorch_lightning as pl
from pytorch_lightning.callbacks import StochasticWeightAveraging
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything
import albumentations as A
from albumentations.pytorch import ToTensorV2

from lib import models
from lib.dataset import CustomDataModule
from lib.utils import initialize_n25, split_data, initialize, get_labels


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Training Model')
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--encoding', type=int, default=111)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--use_aug', action='store_true')
    parser.add_argument('--use_sch', action='store_true')
    parser.add_argument('--use_swa', action='store_true')
    args = parser.parse_args()
    return args


def get_train_transforms(height: int, width: int) -> Sequence[Callable]:
    return A.Compose([
        A.Resize(height=height, width=width),
        A.CLAHE(p=0.2),
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


def get_valid_transforms(height: int, width: int) -> Sequence[Callable]:
    return A.Compose([
        A.Resize(height=height, width=width),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    
def split_kfold(
    k: int = 5, seed: int = 42, root_path: str = 'data', save_name: str = 'kfold.csv'
) -> None:
    train_path = os.path.join(root_path, 'train')
    idxs = list(range(len(os.listdir(train_path))))
    
    train_jpg = sorted(glob(os.path.join(train_path, '*', '*.jpg')))
    train_json = sorted(glob(os.path.join(train_path, '*', '*.json')))
    
    labels = get_labels(train_json)
    
    df = pd.DataFrame({'id': idxs})
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for fold, (_, valid_idx) in enumerate(skf.split(train_jpg, labels)):
        df.loc[valid_idx, 'kfold'] = int(fold)
        
    print(df['kfold'].value_counts())
    save_path = os.path.join(root_path, save_name)
    df.to_csv(save_path, index=False)


def train_base(
    model_name: str, 
    args: argparse.ArgumentParser, 
    csv_feature_dict: Dict[str, List[float]], 
    label_encoder: Dict[str, int], 
    seed: int = 42
) -> None:
    """
    Use for model trained image and time series.
    """
    train_data, val_data = split_data(seed=seed, mode='train')
    
    if args.use_aug:
        train_transforms = get_train_transforms(args.height, args.width)
        model_name += '-aug'
    else:
        train_transforms = get_valid_transforms(args.height, args.width)

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
    
    if args.use_sch:
        model_name += '-sch'
    
    model = models.__dict__[args.model_name](
        max_len=24*6, 
        embedding_dim=512, 
        num_features=len(csv_feature_dict), 
        class_n=len(label_encoder), 
        rate=args.dropout_rate,
        learning_rate=args.lr,
        max_epochs=args.max_epochs,
        use_sch=args.use_sch
    )
    
    progress_bar = RichProgressBar()
    callbacks = [progress_bar]
    
    if args.use_swa:    
        weight_averaging = StochasticWeightAveraging()
        callbacks.append(weight_averaging)
        model_name += '-swa'
        
    ckpt_path = f'./weights/{model_name}/'
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
        
    checkpoint = ModelCheckpoint(
        monitor='val_score',
        dirpath=ckpt_path,
        filename='{epoch}-{val_score:.3f}',
        save_top_k=5,
        mode='max',
        save_weights_only=True,
    )
    callbacks.append(checkpoint)
        
    gpus = list(map(int, args.gpus.split(',')))

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        gpus=gpus,
        strategy=DDPPlugin(find_unused_parameters=False),
        # strategy=DDPPlugin(find_unused_parameters=True),
        precision=16,
        callbacks=callbacks,
        log_every_n_steps=5,
    )

    trainer.fit(model, data_module)
    
    
def train_fold(
    model_name: str, 
    fold: int, 
    args: argparse.ArgumentParser, 
    csv_feature_dict: Dict[str, List[float]], 
    label_encoder: Dict[str, int]
) -> None:
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
        
    if args.use_aug:
        train_transforms = get_train_transforms(args.height, args.width)
        model_name += '-aug'
    else:
        train_transforms = get_valid_transforms(args.height, args.width)

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
    
    if args.use_sch:
        model_name += '-sch'
    
    model = models.__dict__[args.model_name](
        max_len=24*6, 
        embedding_dim=512, 
        num_features=len(csv_feature_dict), 
        class_n=len(label_encoder), 
        rate=args.dropout_rate,
        learning_rate=args.lr,
        max_epochs=args.max_epochs,
        use_sch=args.use_sch
    )
    
    progress_bar = RichProgressBar()
    callbacks = [progress_bar]
    
    if args.use_swa:    
        weight_averaging = StochasticWeightAveraging()
        callbacks.append(weight_averaging)
        model_name += '-swa'
        
    ckpt_path = f'./weights/{model_name}/'
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
        
    checkpoint = ModelCheckpoint(
        monitor='val_score',
        dirpath=ckpt_path,
        filename='{epoch}-{val_score:.3f}',
        save_top_k=-1,
        mode='max',
        save_weights_only=True,
    )
    callbacks.append(checkpoint)
        
    gpus = list(map(int, args.gpus.split(',')))

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        gpus=gpus,
        strategy=DDPPlugin(find_unused_parameters=False),
        # strategy=DDPPlugin(find_unused_parameters=True),
        precision=16,
        callbacks=callbacks,
        log_every_n_steps=5,
    )

    trainer.fit(model, data_module)


def main() -> None:
    seed = 42
    seed_everything(seed)
    
    args = get_args()
    
    if args.encoding == 25:
        csv_feature_dict, label_encoder, _ = initialize_n25()
    elif args.encoding == 111:
        csv_feature_dict, label_encoder, _ = initialize()
    else:
        raise Exception("encoding parameter must be '25' or '111'.")
    
    k = 5
    split_kfold(k=k, seed=seed)

    for fold in range(k):
        train_fold(
            f'{args.model_name}-w{args.width}-h{args.height}-f{fold}', 
            fold, args, csv_feature_dict, label_encoder
        )


if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        print(err)

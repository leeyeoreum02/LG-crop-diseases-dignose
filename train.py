import os
import argparse
# from functools import partial

from pytorch_lightning.plugins import DDPPlugin
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything
import albumentations as A
from albumentations.pytorch import ToTensorV2

from lib.model_effnet import Effnetb02LSTMModel, Effnetb32LSTMModel
from lib.model_effnet import Effnetb72LSTMModel, Effnetb7NS2LSTMModel
from lib.dataset import CustomDataModule
from lib.utils import split_data, initialize


def get_args():
    parser = argparse.ArgumentParser(description='Training Effnetb02LSTM')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    args = parser.parse_args()
    return args


def get_train_transforms(image_size):
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_valid_transforms(image_size):
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def train(model_name, args, csv_feature_dict, label_encoder, seed=42):
    train_data, val_data = split_data(seed=seed, mode='train')
    
    train_transforms = get_train_transforms(args.image_size)
    val_transforms = get_valid_transforms(args.image_size)
    
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


def main():
    seed = 42
    seed_everything(seed)
    
    csv_feature_dict, label_encoder, _ = initialize()
    
    args = get_args()
    
    train('effnetb7ns-lstm', args, csv_feature_dict, label_encoder, seed)


if __name__ == '__main__':
    main()

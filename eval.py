import os
import argparse
from typing import List

import numpy as np
import pandas as pd

import torch
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.utilities.seed import seed_everything
import albumentations as A
from albumentations.pytorch import ToTensorV2
import ttach as tta

from lib.model_effnet import Effnetb02LSTMModel, Effnetb32LSTMModel, Effnetb7NSPlus2LSTMTTAModel
from lib.model_effnet import Effnetb72LSTMModel, Effnetb7NS2LSTMModel
from lib.model_effnet import Effnetb7NS, Effnetb7NSPlus2LSTMModel
from lib.dataset import CustomDataModule, CustomDataModuleV2
from lib.utils import split_data, initialize


def get_args():
    parser = argparse.ArgumentParser(description='Evaluating Effnet')
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=32)
    args = parser.parse_args()
    return args


def get_predict_transforms(height, width):
    return A.Compose([
        A.Resize(height=height, width=width),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    
def get_tta_transforms():
    return tta.Compose([
        tta.HorizontalFlip(),
        tta.Rotate90(angles=[90]),
        tta.Multiply(factors=[0.9, 1.1]),        
    ])
    
    
def voting_folds(submit_dir, submit_paths: List[os.PathLike], save_name):
    submit = pd.read_csv(submit_paths[0])
    for submit_path in submit_paths[1:]:
        label = pd.read_csv(submit_path)[['label']]
        submit = pd.concat([submit, label], axis=1)
    submit['majority'] = submit.mode(axis=1)[0]
    submit_path = os.path.join(submit_dir, f'middle_{save_name}')
    submit.to_csv(submit_path, index=False)
    # submit = submit.iloc[:, [0, -1]]
    # submit.rename(columns={'majority': 'label'}, inplace=True)
    
    # submit_path = os.path.join(submit_dir, save_name)
    # submit.to_csv(submit_path, index=False)
    
    
def get_submission(outputs, save_dir, save_filename, label_decoder):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    outputs = [o.detach().cpu().numpy() for batch in outputs
                                        for o in batch]
    preds = np.array([label_decoder[int(val)] for val in outputs])
    
    submission = pd.read_csv('data/sample_submission.csv')
    submission['label'] = preds
    
    save_file_path = os.path.join(save_dir, save_filename)
    
    submission.to_csv(save_file_path, index=False)


def eval(
    ckpt_path, 
    args, 
    csv_feature_dict, 
    label_encoder, 
    label_decoder,
    submit_save_dir='submissions',
    submit_save_name='baseline_submission.csv',
):
    test_data = split_data(mode='test')
    
    predict_transforms = get_predict_transforms(args.height, args.width)
    
    data_module = CustomDataModule(
        test=test_data,
        csv_feature_dict=csv_feature_dict,
        label_encoder=label_encoder,
        predict_transforms=predict_transforms,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )

    # model = Effnetb02LSTMModel(
    #     max_len=24*6, 
    #     embedding_dim=512, 
    #     num_features=len(csv_feature_dict), 
    #     class_n=len(label_encoder), 
    # )

    # model = Effnetb72LSTMModel(
    #     max_len=24*6, 
    #     embedding_dim=512, 
    #     num_features=len(csv_feature_dict), 
    #     class_n=len(label_encoder), 
    # )
    
    # model = Effnetb7NS2LSTMModel(
    #     max_len=24*6, 
    #     embedding_dim=512, 
    #     num_features=len(csv_feature_dict), 
    #     class_n=len(label_encoder),
    # )
    
    model = Effnetb7NSPlus2LSTMModel(
        max_len=24*6, 
        embedding_dim=512, 
        num_features=len(csv_feature_dict), 
        class_n=len(label_encoder),
    )
    
    # model = Effnetb7NSPlus2LSTMTTAModel(
    #     max_len=24*6, 
    #     embedding_dim=512, 
    #     num_features=len(csv_feature_dict), 
    #     class_n=len(label_encoder),
    #     tta_transforms=get_tta_transforms(),
    # )

    trainer = pl.Trainer(
        # gpus=[0, 1, 2, 3],
        gpus=1,
        # strategy=DDPPlugin(find_unused_parameters=False),
        precision=16,
    )

    ckpt = torch.load(ckpt_path, map_location='cuda:0')
    model.load_state_dict(ckpt['state_dict'])

    outputs = trainer.predict(model, data_module)

    get_submission(outputs, submit_save_dir, submit_save_name, label_decoder)
    
    
def eval_v2(
    ckpt_path, 
    args,
    submit_save_dir='submissions',
    submit_save_name='baseline_submission.csv',
):
    label_decoder = {
        0: '1_00_0', 1: '2_00_0', 2: '2_a5_2', 3: '3_00_0', 4: '3_a9_1', 
        5: '3_a9_2', 6: '3_a9_3', 7: '3_b3_1', 8: '3_b6_1', 9: '3_b7_1', 
        10: '3_b8_1', 11: '4_00_0', 12: '5_00_0', 13: '5_a7_2', 14: '5_b6_1', 
        15: '5_b7_1', 16: '5_b8_1', 17: '6_00_0', 18: '6_a11_1', 19: '6_a11_2',
        20: '6_a12_1', 21: '6_a12_2', 22: '6_b4_1', 23: '6_b4_3', 24: '6_b5_1',
    }
    
    predict_transforms = get_predict_transforms(args.height, args.width)
    
    data_module = CustomDataModuleV2(
        root_path='data',
        predict_transforms=predict_transforms,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )
    
    model = Effnetb7NS()
        
    trainer = pl.Trainer(
        # gpus=[0, 1, 2, 3],
        gpus=[1],
        strategy=DDPPlugin(find_unused_parameters=False),
        precision=16,
    )

    # ckpt = torch.load(ckpt_path, map_location='cuda:1')
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['state_dict'])

    outputs = trainer.predict(model, data_module)
    
    get_submission(outputs, submit_save_dir, submit_save_name, label_decoder)


def main():
    seed = 42
    seed_everything(seed)
    
    ckpt_dir = 'weights/effnetb7nsplus-lstm-w512-h512-f0-aug-sch'
    ckpt_name = 'epoch=94-val_score=0.93.ckpt'
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    
    csv_feature_dict, label_encoder, label_decoder = initialize()
    
    args = get_args()
    
    submit_save_name = 'effnetb7nsplus-lstm-w512-h512-f0-aug-sch-e93-tw600-th600.csv'
    
    # eval(
    #     ckpt_path, args, csv_feature_dict, label_encoder, label_decoder,
    #     submit_save_name=submit_save_name
    # )
    # eval_v2(ckpt_path, args, submit_save_name=submit_save_name)
    
    submit_dir = 'submissions'
    submit_paths = [
        'submissions\effnetb7ns-lstm-w512-h512-f1-aug-e38.csv',
        'submissions\effnetb7nsplus-lstm-w512-h512-f0-aug-sch-e93-tw600-th600.csv',
        'submissions\effnetb7nsplus-lstm-w512-h512-f1-aug-sch-e55-tw512-th512.csv',
        'submissions\effnetb7nsplus-lstm-w512-h512-f1-aug-sch-e55-tw600-th600.csv'
    ]
    save_name = 'best-4fold.csv'
    
    voting_folds(submit_dir, submit_paths, save_name)


if __name__ == '__main__':
    main()

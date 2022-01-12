import os
import argparse

import numpy as np
import pandas as pd

import torch
import pytorch_lightning as pl
# from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.utilities.seed import seed_everything

from lib.model_effnet import Effnetb02LSTMModel, Effnetb72LSTMModel
from lib.dataset import CustomDataModule
from lib.utils import split_data, initialize


def get_args():
    parser = argparse.ArgumentParser(description='Evaluating Effnetb02LSTM')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=32)
    args = parser.parse_args()
    return args
    
    
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
    
    data_module = CustomDataModule(
        test=test_data,
        csv_feature_dict=csv_feature_dict,
        label_encoder=label_encoder,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )

    # model = Effnetb02LSTMModel(
    #     max_len=24*6, 
    #     embedding_dim=512, 
    #     num_features=len(csv_feature_dict), 
    #     class_n=len(label_encoder), 
    # )

    model = Effnetb72LSTMModel(
        max_len=24*6, 
        embedding_dim=512, 
        num_features=len(csv_feature_dict), 
        class_n=len(label_encoder), 
    )

    trainer = pl.Trainer(
        # gpus=[0, 1, 2, 3],
        gpus=1,
        # strategy=DDPPlugin(find_unused_parameters=False),
        precision=16,
    )

    ckpt = torch.load(ckpt_path, map_location='cuda:0')
    model.load_state_dict(ckpt['state_dict'])
    # ckpt = torch.load("./weights/trained_effdet.pth", map_location='cuda:0')
    # model.load_state_dict(ckpt)

    outputs = trainer.predict(model, data_module)

    get_submission(outputs, submit_save_dir, submit_save_name, label_decoder)


def main():
    seed = 42
    seed_everything(seed)
    
    ckpt_dir = 'weights/effnetb7-lstm'
    ckpt_name = 'epoch=38-val_score=0.87.ckpt'
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    
    csv_feature_dict, label_encoder, label_decoder = initialize()
    
    args = get_args()
    
    submit_save_name = 'effnetb72lstm-e38.csv'
    
    eval(
        ckpt_path, args, csv_feature_dict, label_encoder, label_decoder,
        submit_save_name=submit_save_name
    )


if __name__ == '__main__':
    main()

import os
import json
from glob import glob

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold


def get_labels(train_json):
    crops = []
    diseases = []
    risks = []
    labels = []
    
    for i in range(len(train_json)):
        with open(train_json[i], 'r') as f:
            sample = json.load(f)
            crop = sample['annotations']['crop']
            disease = sample['annotations']['disease']
            risk = sample['annotations']['risk']
            label = f"{crop}_{disease}_{risk}"
        
            crops.append(crop)
            diseases.append(disease)
            risks.append(risk)
            labels.append(label)
            
    label_unique = sorted(np.unique(labels))
    label_unique = {key: value for key, value in zip(label_unique, range(len(label_unique)))}
    # print(label_unique)

    labels = [label_unique[k] for k in labels]
    
    return labels


def split_data(split_rate=0.2, seed=42, mode='train'):
    """
    Use for model trained image and time series.
    """
    if mode == 'train':
        train = sorted(glob('data/train/*'))
        
        labelsss = pd.read_csv('data/train.csv')['label']
        train, val = train_test_split(
            train, test_size=split_rate, random_state=seed, stratify=labelsss)
        
        return train, val
    elif mode == 'test':
        test = sorted(glob('data/test/*'))

        return test


def initialize():
    csv_feature_dict = {
        '내부 온도 1 평균': [3.4, 47.3],
        '내부 온도 1 최고': [3.4, 47.6],
        '내부 온도 1 최저': [3.3, 47.0],
        '내부 습도 1 평균': [23.7, 100.0],
        '내부 습도 1 최고': [25.9, 100.0],
        '내부 습도 1 최저': [0.0, 100.0],
        '내부 이슬점 평균': [0.1, 34.5],
        '내부 이슬점 최고': [0.2, 34.7],
        '내부 이슬점 최저': [0.0, 34.4]
    }
    
    crop = {'1': '딸기', '2': '토마토', '3': '파프리카', '4': '오이', '5': '고추', '6': '시설포도'}
    disease = {
        '1': {
            'a1': '딸기잿빛곰팡이병', 'a2': '딸기흰가루병', 'b1': '냉해피해', 
            'b6': '다량원소결핍 (N)', 'b7': '다량원소결핍 (P)', 'b8': '다량원소결핍 (K)'
        },
        '2': {
            'a5': '토마토흰가루병', 'a6': '토마토잿빛곰팡이병', 'b2': '열과', 'b3': '칼슘결핍',
            'b6': '다량원소결핍 (N)', 'b7': '다량원소결핍 (P)', 'b8': '다량원소결핍 (K)'
        },
        '3': {
            'a9': '파프리카흰가루병', 'a10': '파프리카잘록병', 'b3': '칼슘결핍', 
            'b6': '다량원소결핍 (N)', 'b7': '다량원소결핍 (P)', 'b8': '다량원소결핍 (K)'
        },
        '4': {
            'a3': '오이노균병', 'a4': '오이흰가루병', 'b1': '냉해피해', 
            'b6': '다량원소결핍 (N)', 'b7': '다량원소결핍 (P)', 'b8': '다량원소결핍 (K)' 
        },
        '5': {
            'a7': '고추탄저병', 'a8': '고추흰가루병', 'b3': '칼슘결핍', 
            'b6': '다량원소결핍 (N)', 'b7': '다량원소결핍 (P)', 'b8': '다량원소결핍 (K)'
        },
        '6': {'a11': '시설포도탄저병', 'a12': '시설포도노균병', 'b4': '일소피해', 'b5': '축과병'}
    }
    risk = {'1': '초기', '2': '중기', '3': '말기'}
    
    label_description = {}
    for key, value in disease.items():
        label_description[f'{key}_00_0'] = f'{crop[key]}_정상'
        for disease_code in value:
            for risk_code in risk:
                label = f'{key}_{disease_code}_{risk_code}'
                label_description[label] = f'{crop[key]}_{disease[key][disease_code]}_{risk[risk_code]}'
                
    label_encoder = {key: idx for idx, key in enumerate(label_description)}
    # print(len(label_encoder))
    label_decoder = {val: key for key, val in label_encoder.items()}
    
    return csv_feature_dict, label_encoder, label_decoder


def split_kfold(k=5, seed=42, root_path='data', save_name='kfold.csv'):
    """
    Use for model trained only image.
    """
    train_path = os.path.join(root_path, 'train')
    idxs = list(range(len(os.listdir(train_path))))
    
    train_jpg = sorted(glob(os.path.join(train_path, '*', '*.jpg')))
    train_json = sorted(glob(os.path.join(train_path, '*', '*.json')))
    
    labels = get_labels(train_json)
    
    df = pd.DataFrame({'id': idxs})
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for fold, (_, valid_idx) in enumerate(skf.split(train_jpg, labels)):
        df.loc[valid_idx, 'kfold'] = int(fold)
        
    # print(df['kfold'].value_counts())
    save_path = os.path.join(root_path, save_name)
    df.to_csv(save_path, index=False)

from glob import glob

from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(split_rate=0.2, seed=42, mode='train'):
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
    csv_features = [
        '내부 온도 1 평균', '내부 온도 1 최고', '내부 온도 1 최저', '내부 습도 1 평균', '내부 습도 1 최고', 
        '내부 습도 1 최저', '내부 이슬점 평균', '내부 이슬점 최고', '내부 이슬점 최저'
    ]
    csv_files = sorted(glob('data/train/*/*.csv'))

    temp_csv = pd.read_csv(csv_files[0])[csv_features]
    max_arr, min_arr = temp_csv.max().to_numpy(), temp_csv.min().to_numpy()

    # feature 별 최대값, 최솟값 계산
    for csv in tqdm(csv_files[1:]):
        temp_csv = pd.read_csv(csv)[csv_features]
        temp_csv = temp_csv.replace('-', np.nan).dropna()
        if len(temp_csv) == 0:
            continue
        temp_csv = temp_csv.astype(float)
        temp_max, temp_min = temp_csv.max().to_numpy(), temp_csv.min().to_numpy()
        max_arr = np.max([max_arr, temp_max], axis=0)
        min_arr = np.min([min_arr, temp_min], axis=0)
    csv_feature_dict = {csv_features[i]: [min_arr[i], max_arr[i]] for i in range(len(csv_features))}
    
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
    label_decoder = {val: key for key, val in label_encoder.items()}
    
    return csv_feature_dict, label_encoder, label_decoder

import os
from typing import List, Dict

import pandas as pd
from tqdm import tqdm

from lib.utils import initialize, initialize_n25


def voting_folds(
    submit_dir: os.PathLike, submit_paths: List[os.PathLike], save_name: str
) -> None:
    submit = pd.read_csv(submit_paths[0])
    for submit_path in tqdm(submit_paths[1:]):
        label = pd.read_csv(submit_path)[['label']]
        submit = pd.concat([submit, label], axis=1)
    submit['majority'] = submit.iloc[:, 1:].mode(axis=1)[0]
    submit_path = os.path.join(submit_dir, f'middle_{save_name}')
    submit.to_csv(submit_path, index=False)
    submit = submit.iloc[:, [0, -1]]
    submit.rename(columns={'majority': 'label'}, inplace=True)
    
    submit_path = os.path.join(submit_dir, save_name)
    submit.to_csv(submit_path, index=False)
    
    
def sum_folds(
    submit_dir: os.PathLike, 
    submit_paths: List[os.PathLike], 
    save_name: str, 
    label_decoder: Dict[int, str] 
) -> None:
    submit = pd.read_csv(submit_paths[0])
    for submit_path in tqdm(submit_paths[1:]):
        onehot = pd.read_csv(submit_path).iloc[:, 1:]
        onehot = submit.iloc[:, 1:] + onehot
        submit.iloc[:, 1:] = onehot
    submit_path = os.path.join(submit_dir, f'middle_{save_name}')
    submit.to_csv(submit_path, index=False)
    label = submit.iloc[:, 1:].idxmax(axis=1)
    label = pd.Series([label_decoder[int(val)] for val in label])
    submit = pd.concat([submit[['image']], label], axis=1)
    submit.rename(columns={0: 'label'}, inplace=True)
    submit_path = os.path.join(submit_dir, save_name)
    submit.to_csv(submit_path, index=False)
    
    
def get_mean_sum_folds(
    submit_dir: os.PathLike, 
    save_name: str, 
    label_decoder: Dict[int, str], 
    mean1: os.PathLike, 
    mean2: os.PathLike, 
    sum2: os.PathLike, 
    sum3: os.PathLike
) -> None:
    tmp = pd.read_csv(mean1)
    tmp_list = [pd.read_csv(submit_path).iloc[:, 1:] for submit_path in tqdm([mean1, mean2, sum2, sum3])]
    
    onehot = (tmp_list[0] + tmp_list[1]) / 2 + tmp_list[2] + tmp_list[3]
    submit = pd.concat([tmp[['image']], onehot], axis=1)
    submit_path = os.path.join(submit_dir, f'middle_{save_name}')
    submit.to_csv(submit_path, index=False)
    
    label = submit.iloc[:, 1:].idxmax(axis=1)
    label = pd.Series([label_decoder[int(val)] for val in label])
    submit = pd.concat([submit[['image']], label], axis=1)
    submit.rename(columns={0: 'label'}, inplace=True)
    submit_path = os.path.join(submit_dir, save_name)
    submit.to_csv(submit_path, index=False)
    
    
def get_weighted_average_folds(
    submit_paths: List[os.PathLike], 
    weights: List[float], 
    submit_dir: os.PathLike, 
    save_name: str, 
    label_encoder: Dict[str, int], 
    label_decoder: Dict[int, str]
) -> None:
    assert len(submit_paths) == len(weights)
    
    submit = pd.read_csv(submit_paths[0])
    submit[['label']] = submit[['label']].applymap(lambda x: label_encoder[x]) * weights[0]
    for submit_path, weight in tqdm(list(zip(submit_paths[1:], weights[1:]))):
        tmp = pd.read_csv(submit_path)[['label']].applymap(lambda x: label_encoder[x]) * weight
        submit[['label']] += tmp
    submit[['label']] /= len(submit_paths)
    
    submit_path = os.path.join(submit_dir, f'middle_{save_name}')
    submit.to_csv(submit_path, index=False)
    
    submit[['label']] = submit[['label']].applymap(lambda x: label_decoder[round(x)])
    print(submit)
    submit_path = os.path.join(submit_dir, save_name)
    submit.to_csv(submit_path, index=False)
    
    
def recover_best() -> None:
    submit_dir = 'submissions'
    _, _, label_decoder_111 = initialize()
    _, _, label_decoder_25 = initialize_n25()
    
    effnet_stage1_folds = [
        'submissions/effnetb7nsplus-lstm-w384-h384-f0-aug-sch-e95-tw512-th512-onehot.csv',
        'submissions/effnetb7nsplus-lstm-w384-h384-f1-aug-sch-swa-e86-tw512-th512-onehot.csv',
        'submissions/effnetb7nsplus-lstm-w384-h384-f2-aug-sch-swa-e97-tw512-th512-onehot.csv',
        'submissions/effnetb7nsplus-lstm-w384-h384-f3-aug-sch-e87-tw512-th512-onehot.csv',
        'submissions/effnetb7nsplus-lstm-w384-h384-f4-aug-sch-swa-e97-tw512-th512-onehot.csv',
    ]
    sum_folds(submit_dir, effnet_stage1_folds, 'effnet_stage1.csv', label_decoder_111)
    
    effnet_stage2_folds = [
        'submissions/efficientnet_b7_ns_lstm-w512-h512-f1-aug-e32-tw512-th512-onehot.csv',
        'submissions/effnetb7nsplus-lstm-w512-h512-f0-aug-sch-e94-tw600-th600-onehot.csv',
        'submissions/effnetb7nsplus-lstm-w512-h512-f1-aug-sch-e55-tw512-th512-onehot.csv',
        'submissions/effnetb7nsplus-lstm-w512-h512-f1-aug-sch-e55-tw600-th600-onehot.csv',
    ]
    sum_folds(submit_dir, effnet_stage2_folds, 'effnet_stage2.csv', label_decoder_111)
    
    effnet_stage3_folds = [
        'submissions/effnetb7nsplus-lstm-w384-h384-f0-aug-sch-e95-tw512-th512-onehot.csv',
        'submissions/effnetb7nsplus-lstm-w384-h384-f1-aug-sch-swa-e86-tw512-th512-onehot.csv',
        'submissions/effnetb7nsplus-lstm-w384-h384-f2-aug-sch-swa-e97-tw512-th512-onehot.csv',
        'submissions/effnetb7nsplus-lstm-w384-h384-f3-aug-sch-e87-tw512-th512-onehot.csv',
        'submissions/effnetb7nsplus-lstm-w384-h384-f4-aug-sch-swa-e97-tw512-th512-onehot.csv',
        'submissions/effnetb2nsplus-lstm-w384-h384-f1-aug-sch-swa-e72-tw384-th384-onehot.csv',
        'submissions/effnetb2nsplus-lstm-w384-h384-f4-aug-sch-swa-s777-e74-tw384-th384-onehot.csv',
    ]
    sum_folds(submit_dir, effnet_stage3_folds, 'effnet_stage3.csv', label_decoder_111)
    
    beit_stage_folds = [
        'submissions/beitlarge384p16-lstm-w384-h384-f0-aug-sch-swa-e81-tw384-th384-onehot.csv',
        'submissions/beitlarge384p16-lstm-w384-h384-f1-aug-sch-swa-e86-tw384-th384-onehot.csv',
        'submissions/beitlarge384p16-lstm-w384-h384-f2-aug-sch-swa-e71-tw384-th384-onehot.csv',
        'submissions/beitlarge384p16-lstm-w384-h384-f3-aug-sch-swa-e86-tw384-th384-onehot.csv',
        'submissions/beitlarge384p16-lstm-w384-h384-f4-aug-sch-swa-e99-tw384-th384-onehot.csv',
    ]
    sum_folds(submit_dir, beit_stage_folds, 'beit_stage.csv', label_decoder_25)
    
    onehot_ensembles = [
        os.path.join(submit_dir, 'middle_effnet_stage1.csv'), 
        os.path.join(submit_dir, 'middle_effnet_stage2.csv'), 
        os.path.join(submit_dir, 'middle_effnet_stage3.csv'),
    ]
    sum_folds(submit_dir, onehot_ensembles, 'effnet_sum_ensemble.csv', label_decoder_111)
    ensembles = [
        os.path.join(submit_dir, 'effnet_stage1.csv'), 
        os.path.join(submit_dir, 'effnet_stage2.csv'), 
        os.path.join(submit_dir, 'effnet_stage3.csv'),
    ]
    voting_folds(submit_dir, ensembles, 'effnet_voting_ensemble.csv')
    
    ensembles.append(os.path.join(submit_dir, 'beit_stage.csv'))
    voting_folds(submit_dir, ensembles, 'final.csv') 
    
    
def main() -> None:
    recover_best()


if __name__ == '__main__':
    main()

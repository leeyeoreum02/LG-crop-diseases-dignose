# Dacon 농업 환경 변화에 따른 작물 병해 진단 AI 경진대회

## Quick Start
- Setup
```shell
git clone https://github.com/leeyeoreum02/LG-crop-diseases-dignose.git
cd LG-crop-disease-dignose
conda env create -f environment.yaml
conda activate lg-crop
```
- Train
```shell
sh train.sh
```
- Evaluation(recover submission)
```shell
sh eval.sh
```

- Weight download


    zip 파일 다운로드 후 ./weights 디렉토리에 압축해제 (디렉토리 없으면 따로 생성해야 함)

## Weights
- [Download weights zip file (.ckpt)](https://drive.google.com/file/d/1cnUXkSkc9aENuPJvpk8dJe25VQ1CLPwJ/view?usp=sharing)

## Summary
- Strong augmentation


- EfficientNet + Noisy Student (default)
- Custom EfficientNet + Noisy Student (drop_path_rate: 0.4, dropout_rate: 0.5)
- Beit Large P16
- Baseline LSTM
- Ensemble multi-scale model: sum, voting
- Test time augmentation: HorizontalFlip, Rotate90, Multiply

## Data Preprocessing
- Strong augmentation

    CLAHE, RandomBrightnessContrast, ColorJitter, RGBShift, RandomSnow, RandomCrop, HorizontalFlip, VerticalFlip, Rotate, RandomRotate90

## Model
- 5 folds, stratified-kfold


- Optimizer: 
    - Adam with initial LR 5e-4 for Noisy Student+LSTM


    - Adam with initial LR 5e-4 for Beit Large P16+LSTM
- LR scheduler: linear-warmup-cosine-annealing
- Stocastic weight averaging training
- Mixed precision training with pytorch lightning

## Using Pretrained Model
Model from timm, torchvision


- EfficientNet-B7 + Noisy Student + LSTM, image size 512: Fold 1


- Custom EfficientNet-B2 + Noisy Student + LSTM, image size 384: Fold 0,1
- Custom EfficientNet-B7 + Noisy Student + LSTM, image size 384, test image size: 512: Fold 0,1,2,3,4
- Custom EfficientNet-B7 + Noisy Student + LSTM, image size 512: Fold 1
- Custom EfficientNet-B7 + Noisy Student + LSTM, image size 512, test image size 600: Fold 0,1
- Beit Large P16 + LSTM, image size 384: Fold 0,1,2,3,4

## Environment
- OS: Ubuntu 18.04.4 LTS (GNU/Linux 4.15.0-162-generic x86_64)


- Environment: Anaconda 4.10.3

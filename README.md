# Dacon 농업 환경 변화에 따른 작물 병해 진단 AI 경진대회

## Summary
- Heavy augmentation
- EfficientNet + Noisy Student
- Baseline LSTM

### Heavy augmentation
- CLAHE, RandomBrightnessContrast, ColorJitter, RGBShift, RandomSnow, 
- RandomCrop, 
- HorizontalFlip, VerticalFlip, 
- Rotate, RandomRotate90

### Model
- 5 folds
- Optimizer: Adam with initial LR 5e-4 for Noisy Student+LSTM
- Mixed precision training with pytorch lightning

### Environment
- OS: Ubuntu 18.04.4 LTS (GNU/Linux 4.15.0-162-generic x86_64)
- Environment: Anaconda 4.10.3

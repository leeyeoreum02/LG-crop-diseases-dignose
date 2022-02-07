# python train.py --model_name vit_large_384_p16 --encoding 25 --width 384 --height 384 --batch_size 14 --max_epochs 100 --num_workers 16 --gpus 1,2 --use_aug --use_sch --use_swa
# python train.py --model_name efficientnet_b7_ns_plus_focal_lstm --encoding 111 --width 384 --height 384 --batch_size 32 --max_epochs 100 --num_workers 16 --gpus 1,2,3 --use_aug --use_sch --use_swa
python train.py --model_name efficientnet_b7_ns_lstm --encoding 111 --width 512 --height 512 --batch_size 16 --max_epochs 39 --num_workers 16 --gpus 1,2,3 --use_aug
# python train.py --model_name efficientnet_b2_ns_plus_focal_lstm --encoding 111 --width 384 --height 384 --batch_size 64 --max_epochs 75 --num_workers 16 --gpus 1,2,3 --use_aug --use_sch --use_swa

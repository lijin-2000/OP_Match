CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar10 --num-labeled 50 --out /root/nas-public-linkdata/lijin/OP_match_res/cifar10/yuan_OP_50_grad --arch wideresnet --lambda_oem 0.1 --lambda_socr 0.5 \
--batch-size 64 --lr 0.03 --expand-labels --seed 0 --opt_level O2 --amp --mu 2 --OODSimilar_classes True\











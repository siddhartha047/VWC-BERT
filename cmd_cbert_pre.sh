#!/bin/bash

/bin/hostname

module load anaconda/2020.11
conda activate deep_38 

cd /homes/das90/GNNcodes/GNN-NC/CBERT/

python 1-CBERT-Pretraining.py --pretrained='distilbert-base-uncased' --num_gpus=2 --parallel_mode='ddp' --epochs=30 --batch_size=16 --refresh_rate=200

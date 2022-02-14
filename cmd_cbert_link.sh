#!/bin/bash

/bin/hostname

module load anaconda/2020.11
conda activate deep_38 

cd /homes/das90/GNNcodes/GNN-NC/CBERT/

python 2-CBERT-Link-Prediction.py --pretrained='distilbert-base-uncased' --use_pretrained=True --use_rd=False --checkpointing=True --rand_dataset='dummy'  --performance_mode=False --neg_link=128  --epoch=25 --nodes=1 --num_gpus=2 --batch_size=64

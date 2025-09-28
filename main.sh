#!/bin/bash
nn=ann # ann or snn
model=vgg16 # vgg16 or resnet20
CUDA_VISIBLE_DEVICES=1
python main.py --nn $nn --model $model
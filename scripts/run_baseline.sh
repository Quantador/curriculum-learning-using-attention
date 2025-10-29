#!/usr/bin/env bash
python -m src.train \
  --dataset_cfg configs/dataset.yaml \
  --model_cfg configs/model.yaml \
  --train_cfg configs/train.yaml

#!/usr/bin/env bash
# Straight-Through top-k (hard forward, soft back)
python -m src.train_router \
  --dataset_cfg configs/dataset.yaml \
  --model_cfg configs/model.yaml \
  --train_cfg configs/train.yaml \
  --router_cfg configs/router.yaml

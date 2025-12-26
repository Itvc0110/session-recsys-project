#!/bin/bash
models=("sasrec" "gru4rec")
datasets=("ml1m" "beauty" "sports" "yelp")  # Add others later

for model in "${models[@]}"; do
  for dataset in "${datasets[@]}"; do
    python scripts/train.py --model $model --dataset $dataset
    python scripts/evaluate.py --model $model --dataset $dataset
  done
done

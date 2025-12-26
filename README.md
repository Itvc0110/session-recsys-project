# Session-RecSys-Project
Project for training/evaluating SASRec and GRU4Rec (extend to CL4SRec later) on ML-1M (extend to Beauty, Sports, Yelp).

## Data Download and Storage
Download MovieLens-1M from https://grouplens.org/datasets/movielens/1m/. Unzip and place ratings.dat in data/raw/ml-1m/ratings.dat. The code expects this path for loading.

## Installation
pip install -r requirements.txt
pip install -e .

## Usage
python scripts/train.py --model sasrec --dataset ml1m
python scripts/evaluate.py --model sasrec --dataset ml1m

## Expected Performance on ML-1M
- SASRec: Hit@10 ~0.82, NDCG@10 ~0.59 (per original paper)
- GRU4Rec: Hit@10 ~0.70-0.75, NDCG@10 ~0.45-0.50 (benchmarks)

## Citations
- SASRec: https://arxiv.org/abs/1808.09781
- GRU4Rec: https://arxiv.org/abs/1511.06939Initialize

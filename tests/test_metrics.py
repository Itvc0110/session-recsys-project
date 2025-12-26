import pytest
import torch
from src.metrics import hit_at_k, ndcg_at_k

def test_hit():
    preds = torch.tensor([[0.9, 0.8, 0.7]])
    labels = torch.tensor([0])
    assert hit_at_k(preds, labels, 3) == 1.0

def test_ndcg():
    preds = torch.tensor([[0.9, 0.8, 0.7]])
    labels = torch.tensor([1])  
    assert ndcg_at_k(preds, labels, 3) > 0

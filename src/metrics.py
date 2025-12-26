import torch

def hit_at_k(preds, labels, k=10):
    """Recall@K: fraction of relevant items in top-k."""
    topk = torch.topk(preds, k).indices
    return (topk == labels.unsqueeze(1)).any(1).float().mean()

def ndcg_at_k(preds, labels, k=10):
    """NDCG@K: normalized discounted cumulative gain."""
    topk = torch.topk(preds, k).indices
    ideal = torch.arange(1, k+1, dtype=torch.float, device=preds.device)
    gains = (topk == labels.unsqueeze(1)).float() / torch.log2(ideal + 1)
    return gains.sum(1).mean()

def mrr_at_k(preds, labels, k=10):
    """MRR@K: mean reciprocal rank."""
    topk = torch.topk(preds, k).indices
    ranks = (topk == labels.unsqueeze(1)).nonzero(as_tuple=True)[1] + 1
    reciprocals = 1.0 / ranks.float()
    return reciprocals.mean()

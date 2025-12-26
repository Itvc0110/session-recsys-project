import torch

def hit_at_k(preds, labels, k=10):
    """Hit@K: 1 if the true item is in top-k predictions."""
    topk_indices = torch.topk(preds, k, dim=1).indices  # (batch_size, k)
    hits = (topk_indices == labels.unsqueeze(1)).any(dim=1).float()  # (batch_size,)
    return hits.mean()

def ndcg_at_k(preds, labels, k=10):
    """NDCG@K with single relevant item."""
    topk_indices = torch.topk(preds, k, dim=1).indices  # (batch_size, k)
    relevance = (topk_indices == labels.unsqueeze(1)).float()  # (batch_size, k)
    discounts = 1.0 / torch.log2(torch.arange(1, k+1, dtype=torch.float, device=preds.device) + 1)
    dcg = (relevance * discounts).sum(dim=1)  # (batch_size,)
    idcg = 1.0
    return (dcg / idcg).mean()

def mrr_at_k(preds, labels, k=10):
    """MRR@K: reciprocal rank of the first relevant item."""
    topk_indices = torch.topk(preds, k, dim=1).indices  # (batch_size, k)
    matches = (topk_indices == labels.unsqueeze(1)).nonzero(as_tuple=True)
    if len(matches[1]) == 0:
        return torch.tensor(0.0, device=preds.device)  # No hit in top-k
    ranks = matches[1] + 1  # 1-based rank
    reciprocals = 1.0 / ranks.float()
    return reciprocals.mean()

import torch

def hit_at_k(preds, labels, k=10):
    """Hit@K: fraction of samples where the true item is in top-k."""
    # preds: (batch_size, num_items), labels: (batch_size,)
    topk_indices = torch.topk(preds, k, dim=1).indices  # (batch_size, k)
    # Expand labels to (batch_size, k) for comparison
    labels_expanded = labels.unsqueeze(1).expand(-1, k)  # (batch_size, k)
    hits = (topk_indices == labels_expanded).any(dim=1).float()
    return hits.mean()

def ndcg_at_k(preds, labels, k=10):
    """NDCG@K for single relevant item."""
    topk_indices = torch.topk(preds, k, dim=1).indices  # (batch_size, k)
    labels_expanded = labels.unsqueeze(1).expand(-1, k)  # (batch_size, k)
    relevance = (topk_indices == labels_expanded).float()  # (batch_size, k)
    
    # Discounted cumulative gain
    discounts = 1.0 / torch.log2(torch.arange(1, k+1, dtype=torch.float, device=preds.device) + 1)
    dcg = (relevance * discounts).sum(dim=1)  # (batch_size,)
    
    # Ideal DCG = 1 (single relevant item at rank 1)
    idcg = 1.0
    return (dcg / idcg).mean()

def mrr_at_k(preds, labels, k=10):
    """MRR@K: reciprocal rank of the first relevant item."""
    topk_indices = torch.topk(preds, k, dim=1).indices  # (batch_size, k)
    labels_expanded = labels.unsqueeze(1).expand(-1, k)  # (batch_size, k)
    
    # Find the first match per row
    matches = (topk_indices == labels_expanded)
    ranks = matches.nonzero(as_tuple=True)[1] + 1  # 1-based ranks
    if len(ranks) == 0:
        return torch.tensor(0.0, device=preds.device)
    
    reciprocals = 1.0 / ranks.float()
    return reciprocals.mean()

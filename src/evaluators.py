from src.metrics import hit_at_k, ndcg_at_k
import torch

class Evaluator:
    def __init__(self, config):
        self.metrics = config['eval_metrics']
        self.k = 10

    def evaluate(self, dataloader, model):
        results = {}
        all_preds, all_labels = [], []
        for batch in dataloader:
            interaction = {k: v.to(model.device) for k, v in batch.items()}
            preds = model.full_sort_predict(interaction)
            all_preds.append(preds)
            pos_item = interaction['pos_item'].view(-1)  # flatten to (batch_size,)
            all_labels.append(pos_item)
        preds = torch.cat(all_preds)
        labels = torch.cat(all_labels)
    
        # Config-driven: exact match
        for metric in self.metrics:
            if metric == 'Hit@10':
                results[metric] = hit_at_k(preds, labels, self.k)
            elif metric == 'NDCG@10':
                results[metric] = ndcg_at_k(preds, labels, self.k)
            elif metric == 'MRR@10':
                results[metric] = mrr_at_k(preds, labels, self.k)
    
        return results

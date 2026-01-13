from src.metrics import hit_at_k, ndcg_at_k, mrr_at_k
import torch

class Evaluator:
    def __init__(self, config):
        self.metrics = config['eval_metrics']
        self.k_values = config['eval_k_values']

    def evaluate(self, dataloader, model):
        results = {}
        all_preds, all_labels = [], []
        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                interaction = {k: v.to(model.device) for k, v in batch.items()}
                preds = model.full_sort_predict(interaction)
                all_preds.append(preds)
                lengths = interaction['item_seq_len']
                last_pos = interaction['pos_item'].gather(1, (lengths - 1).unsqueeze(1)).squeeze(1)
                all_labels.append(last_pos)
        preds = torch.cat(all_preds)
        labels = torch.cat(all_labels)

        for metric in self.metrics:
            for k in self.k_values:
                key = f"{metric}@{k}"
                if metric == 'Hit':
                    results[key] = hit_at_k(preds, labels, k)
                elif metric == 'NDCG':
                    results[key] = ndcg_at_k(preds, labels, k)
                elif metric == 'MRR':
                    results[key] = mrr_at_k(preds, labels, k)

        return results
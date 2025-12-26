from src.metrics import hit_at_k, ndcg_at_k
import torch

class Evaluator:
    def __init__(self, config):
        self.metrics = config['eval_metrics']
        self.k = 10

    def evaluate(self, dataloader, model):
        results = {}
        all_preds, all_labels = [], []
        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                interaction = {k: v.to(model.device) for k, v in batch.items()}
                # For eval, predict on last position
                interaction['pos_item'] = interaction['pos_item'][:, -1:]
                interaction['neg_item'] = interaction['neg_item'][:, -1:]
                preds = model.full_sort_predict(interaction)
                all_preds.append(preds)
                all_labels.append(interaction['pos_item'].squeeze())
        preds = torch.cat(all_preds)
        labels = torch.cat(all_labels)
        for metric in self.metrics:
            if 'Hit' in metric:
                results[metric] = hit_at_k(preds, labels, self.k)
            elif 'NDCG' in metric:
                results[metric] = ndcg_at_k(preds, labels, self.k)
        return results

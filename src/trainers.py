import torch
from torch.optim import Adam
from src.helpers import early_stopping, save_checkpoint

class Trainer:
    def __init__(self, model, config):
        self.model = model.to(config['device'])
        self.optimizer = Adam(self.model.parameters(), lr=config['learning_rate'])
        self.epochs = config['epochs']
        self.device = config['device']
        self.eval_step = 10
        self.patience = 10
        self.best_score = -float('inf')
        self.valid_metric = 'NDCG@10'  

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        for batch in dataloader:
            interaction = {k: v.to(self.device) for k, v in batch.items()}
            loss = self.model.calculate_loss(interaction)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)

    def valid_epoch(self, dataloader, evaluator):
        self.model.eval()
        with torch.no_grad():
            results = evaluator.evaluate(dataloader, self.model)
        return results[self.valid_metric], results

    def fit(self, train_dataloader, valid_dataloader, evaluator):
        counter = 0
        for epoch in range(self.epochs):
            train_loss = self.train_epoch(train_dataloader)
            if epoch % self.eval_step == 0:
                valid_score, valid_result = self.valid_epoch(valid_dataloader, evaluator)
                if valid_score > self.best_score:
                    self.best_score = valid_score
                    save_checkpoint(self.model, epoch)
                    counter = 0
                else:
                    counter += 1
                    if counter >= self.patience:
                        break
        return self.best_score

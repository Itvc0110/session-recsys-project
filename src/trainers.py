import torch
import logging
import os
from src.helpers import early_stopping, save_checkpoint

os.makedirs('experiments/logs', exist_ok=True)
logging.basicConfig(
    filename='experiments/logs/train.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

class Trainer:
    def __init__(self, model, config):
        self.model = model.to(config['device'])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.epochs = config['epochs']
        self.device = config['device']
        self.eval_step = 10
        self.patience = 10
        self.best_score = -float('inf')
        self.valid_metric = 'NDCG@10'

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0
        for batch in dataloader:
            interaction = {k: v.to(self.device) for k, v in batch.items()}
            loss = self.model.calculate_loss(interaction)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        logging.info(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}")
        print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}")
        return avg_loss

    def valid_epoch(self, dataloader, evaluator):
        self.model.eval()
        with torch.no_grad():
            results = evaluator.evaluate(dataloader, self.model)
        logging.info(f"Valid {self.valid_metric}: {results[self.valid_metric]:.4f} | {results}")
        print(f"Valid {self.valid_metric}: {results[self.valid_metric]:.4f}")
        return results[self.valid_metric], results

    def fit(self, train_dataloader, valid_dataloader, evaluator):
        counter = 0
        last_valid_result = None  # Store the last validation results
        for epoch in range(self.epochs):
            train_loss = self.train_epoch(train_dataloader, epoch)
            if epoch % self.eval_step == 0 or epoch == self.epochs - 1:  # Also check at last epoch
                valid_score, valid_result = self.valid_epoch(valid_dataloader, evaluator)
                last_valid_result = valid_result  # Save the latest results
                if valid_score > self.best_score:
                    self.best_score = valid_score
                    save_checkpoint(self.model, epoch, path=f'experiments/checkpoints/{self.model.__class__.__name__.lower()}_model.pth')
                    counter = 0
                else:
                    counter += 1
                    if counter >= self.patience:
                        logging.info(f"Early stopping at epoch {epoch+1}")
                        print(f"Early stopping at epoch {epoch+1}")
                        break
        return self.best_score, last_valid_result



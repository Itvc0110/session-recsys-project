import torch
import os

def early_stopping(score, best, patience=10):
    if score > best:  
            return True, False
    patience -= 1
    return False, patience <= 0

def save_checkpoint(model, epoch, path='checkpoints/model.pth'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_checkpoint(model, path):
    model.load_state_dict(torch.load(path))

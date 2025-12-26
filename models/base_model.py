import torch.nn as nn

class BaseSequentialModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_items = config['num_items']
        self.embedding_size = config['embedding_size']
        self.max_seq_len = config['max_seq_len']
        self.device = config['device']
        self.loss_type = config['loss_type']

    def forward(self, item_seq, item_seq_len):
        raise NotImplementedError

    def calculate_loss(self, interaction):
        raise NotImplementedError

    def full_sort_predict(self, interaction):
        raise NotImplementedError

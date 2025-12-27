import torch
import torch.nn as nn
from models.base_model import BaseSequentialModel

class GRU4Rec(BaseSequentialModel):
    def __init__(self, config):
        super().__init__(config)
        self.hidden_size = config['hidden_size']

        self.item_embedding = nn.Embedding(self.num_items, self.embedding_size, padding_idx=0)
        self.emb_dropout = nn.Dropout(config['dropout_prob'])
        self.gru_layers = nn.GRU(
            input_size=self.embedding_size, hidden_size=self.hidden_size,
            num_layers=config['num_layers'], bias=False, batch_first=True
        )
        self.dense = nn.Linear(self.hidden_size, self.embedding_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.xavier_normal_(module.weight)
        elif isinstance(module, nn.GRU):
            nn.init.xavier_uniform_(module.weight_hh_l0)
            nn.init.xavier_uniform_(module.weight_ih_l0)

    def gather_indexes(self, output, gather_index):
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        return output.gather(dim=1, index=gather_index).squeeze(1)

    def forward(self, item_seq, item_seq_len):
        item_seq_emb = self.item_embedding(item_seq)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        gru_output, _ = self.gru_layers(item_seq_emb_dropout)
        gru_output = self.dense(gru_output)
        return gru_output

    def calculate_loss(self, interaction):
        item_seq = interaction['item_seq']  # (batch, seq_len)
        item_seq_len = interaction['item_seq_len']  # (batch,)
        pos_item = interaction['pos_item']  # (batch, seq_len)
        neg_item = interaction['neg_item']  # (batch, seq_len)
    
        output = self.forward(item_seq, item_seq_len)  # (batch, seq_len, embed_size)
        pos_emb = self.item_embedding(pos_item)  # (batch, seq_len, embed_size)
        neg_emb = self.item_embedding(neg_item)  # (batch, seq_len, embed_size)
    
        pos_scores = (output * pos_emb).sum(dim=-1)  # (batch, seq_len)
        neg_scores = (output * neg_emb).sum(dim=-1)  # (batch, seq_len)
    
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores))  # (batch, seq_len)
    
        # Mask for padding positions
        mask = (item_seq > 0).float()  # (batch, seq_len)
        loss = loss * mask
    
        # Average over non-padding positions
        return loss.sum() / mask.sum()

    def full_sort_predict(self, interaction):
        item_seq = interaction['item_seq']
        item_seq_len = interaction['item_seq_len']
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores


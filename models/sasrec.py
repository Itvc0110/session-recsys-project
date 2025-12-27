import torch
import torch.nn as nn
import copy
import math
import functools
from data.utils import get_attention_mask
from models.base_model import BaseSequentialModel

class TransformerEncoder(nn.Module):
    def __init__(self, n_layers, n_heads, hidden_size, inner_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps):
        super().__init__()
        layer = TransformerLayer(n_heads, hidden_size, inner_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers
    
class TransformerLayer(nn.Module):
    def __init__(self, n_heads, hidden_size, intermediate_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps)
        self.feed_forward = FeedForward(hidden_size, intermediate_size, hidden_dropout_prob, hidden_act, layer_norm_eps)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.multi_head_attention(hidden_states, attention_mask)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output
    
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps):
        super().__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(f"hidden_size {hidden_size} must be divisible by n_heads {n_heads}")
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_size = hidden_size // n_heads

        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.linear_v = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, input_tensor, attention_mask):
        batch_size = input_tensor.size(0)
        query = self.linear_q(input_tensor)
        key = self.linear_k(input_tensor)
        value = self.linear_v(input_tensor)

        query = query.view(batch_size, -1, self.n_heads, self.head_size).permute(0, 2, 1, 3)
        key = key.view(batch_size, -1, self.n_heads, self.head_size).permute(0, 2, 1, 3)
        value = value.view(batch_size, -1, self.n_heads, self.head_size).permute(0, 2, 1, 3)

        attn_scores = torch.matmul(query, key.transpose(-1, -2)) / (self.head_size ** 0.5)
        attn_scores = attn_scores + attention_mask

        attn_probs = nn.Softmax(dim=-1)(attn_scores)
        attn_probs = self.attn_dropout(attn_probs)
        context = torch.matmul(attn_probs, value).permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.hidden_size)

        output = self.out_proj(context)
        output = self.out_dropout(output)
        output = self.layer_norm(output + input_tensor)
        return output
    
class FeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_dropout_prob, hidden_act, layer_norm_eps):
        super().__init__()
        self.dense_1 = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = self.get_hidden_act(hidden_act)

        self.dense_2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def get_hidden_act(self, act):
        ACT2FN = {
            "gelu": self.gelu,
            "relu": functools.partial(nn.ReLU),
            "swish": self.swish,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
        }
        return ACT2FN[act]

    def gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    
class SASRec(BaseSequentialModel):
    def __init__(self, config):
        super().__init__(config)
        self.heads = config['heads']
        self.inner_size = config['inner_size']
        self.hidden_act = 'gelu'
        self.layer_norm_eps = 1e-12

        self.item_embedding = nn.Embedding(self.num_items, self.embedding_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_len, self.embedding_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=config['num_layers'], n_heads=self.heads, hidden_size=self.embedding_size,
            inner_size=self.inner_size, hidden_dropout_prob=config['dropout_prob'],
            attn_dropout_prob=config['dropout_prob'], hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )
        self.LayerNorm = nn.LayerNorm(self.embedding_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(config['dropout_prob'])

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def gather_indexes(self, output, gather_index):
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        return output.gather(dim=1, index=gather_index).squeeze(1)

    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device).unsqueeze(0)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        return output

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
        output = self.forward(item_seq, item_seq_len)
        # Gather the last position output
        seq_output = output.gather(1, (item_seq_len - 1).unsqueeze(1).unsqueeze(2).expand(-1, -1, output.size(-1))).squeeze(1)  # (batch, embed_size)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        # Mask padding token
        scores[:, 0] = -1e9
        return scores



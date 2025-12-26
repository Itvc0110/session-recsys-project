import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

def pad_collate(batch, padding_value=0, num_items=None, neg_samples=1):
    users, input_seqs, targets = zip(*batch)
    input_seqs = [torch.tensor(seq) for seq in input_seqs]
    targets = [torch.tensor(tgt) for tgt in targets]
    padded_inputs = pad_sequence(input_seqs, batch_first=True, padding_value=padding_value)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=padding_value)
    lengths = torch.tensor([len(seq) for seq in input_seqs])
    
    # Negative samples for BPR (random, avoid positives)
    batch_size, seq_len = padded_inputs.shape
    negs = torch.randint(1, num_items, (batch_size, seq_len))
    for i in range(batch_size):
        for j in range(lengths[i]):
            while negs[i, j] in input_seqs[i] or negs[i, j] == targets[i][j]:
                negs[i, j] = torch.randint(1, num_items, ())
    
    return {
        'user': torch.tensor(users),
        'item_seq': padded_inputs,
        'item_seq_len': lengths,
        'pos_item': padded_targets,
        'neg_item': negs
    }

def create_dataloader(sequences, batch_size, shuffle=True, num_items=None, neg_samples=1):
    dataset = [(user, seq[:-1], seq[1:]) for user, seq in sequences.items() if len(seq) > 1]
    collate_fn = lambda b: pad_collate(b, 0, num_items, neg_samples)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

def get_attention_mask(item_seq):
    attention_mask = (item_seq > 0).long()
    extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
    max_len = attention_mask.shape[-1]
    attn_shape = (1, max_len, max_len)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)
    subsequent_mask = (subsequent_mask == 0).unsqueeze(1).long().to(item_seq.device)
    extended_attention_mask = extended_attention_mask * subsequent_mask
    extended_attention_mask = extended_attention_mask.to(dtype=item_seq.dtype)
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask


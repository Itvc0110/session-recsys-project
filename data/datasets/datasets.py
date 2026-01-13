# Need update

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from data.preprocess import apply_5core_filtering, remove_duplicates, sort_by_timestamp

class SequentialDataset(Dataset):
    def __init__(self, data_path, sep=',', max_seq_len=50, min_interactions=5):  # e.g., 50 for Beauty
        data = pd.read_json(data_path, lines=True)  
        data = data.rename(columns={'reviewerID': 'user_id', 'asin': 'item_id', 'unixReviewTime': 'timestamp'})
        data = data[data['overall'] > 0]  
        data = remove_duplicates(data)
        data = sort_by_timestamp(data)
        data = apply_5core_filtering(data, min_interactions)
        
        self.user_map = {uid: idx for idx, uid in enumerate(data['user_id'].unique())}
        self.item_map = {iid: idx for idx, iid in enumerate(data['item_id'].unique())}
        data['user_id'] = data['user_id'].map(self.user_map)
        data['item_id'] = data['item_id'].map(self.item_map)
        
        self.sequences = {}
        grouped = data.groupby('user_id')
        for user, group in grouped:
            seq = group['item_id'].tolist()[-max_seq_len:]
            self.sequences[user] = seq
        
        self.users = list(self.sequences.keys())
        self.num_users = len(self.user_map)
        self.num_items = len(self.item_map) + 1  
    
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        user = self.users[idx]
        seq = self.sequences[user]
        return user, seq[:-1], seq[1:]  
    
    def split(self):
        train_seqs, valid_seqs, test_seqs = {}, {}, {}
        for user, seq in self.sequences.items():
            if len(seq) < 3: continue
            train_seqs[user] = seq[:-2]
            valid_seqs[user] = seq[:-1]  
            test_seqs[user] = seq  
        return train_seqs, valid_seqs, test_seqs

# data/datasets/base_dataset.py
import pandas as pd
from torch.utils.data import Dataset
from data.preprocess import remove_duplicates, sort_by_timestamp, apply_5core_filtering

class BaseSequentialDataset(Dataset):
    def __init__(self, data_path, sep=',', max_seq_len=None, min_interactions=5, is_json=False):
        if is_json:
            data = pd.read_json(data_path, lines=True)
        else:
            data = pd.read_csv(data_path, sep=sep, engine='python')

        # Rename columns to standard names (override in child classes if needed)
        rename_map = {
            'customer_id': 'user_id', 'reviewerID': 'user_id', 'userId': 'user_id',
            'product_id': 'item_id', 'asin': 'item_id', 'movieId': 'item_id',
            'unixReviewTime': 'timestamp', 'review_date': 'timestamp', 'date': 'timestamp',
            'overall': 'rating', 'star_rating': 'rating', 'stars': 'rating'
        }
        data = data.rename(columns=rename_map)

        # Binarize implicit feedback
        if 'rating' in data.columns:
            data = data[data['rating'] > 0]

        # Keep only needed columns
        data = data[['user_id', 'item_id', 'timestamp']]

        # Preprocessing
        data = remove_duplicates(data)
        data = sort_by_timestamp(data)
        data = apply_5core_filtering(data, min_interactions)

        # ID mapping (0 reserved for padding)
        self.user_map = {uid: idx for idx, uid in enumerate(data['user_id'].unique())}
        self.item_map = {iid: idx + 1 for idx, iid in enumerate(data['item_id'].unique())}
        data['user_id'] = data['user_id'].map(self.user_map)
        data['item_id'] = data['item_id'].map(self.item_map)

        # Build sequences (most recent first)
        self.sequences = {}
        grouped = data.groupby('user_id')
        for user, group in grouped:
            seq = group['item_id'].tolist()
            if max_seq_len is not None:
                seq = seq[-max_seq_len:]
            self.sequences[user] = seq

        self.users = list(self.sequences.keys())
        self.num_users = len(self.user_map)
        self.num_items = len(self.item_map) + 1  # +1 for padding

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        seq = self.sequences[user]
        if len(seq) == 0:
            return user, torch.tensor([]), torch.tensor([])
        return user, seq[:-1], seq[1:]  # input, targets

    def split(self):
        train_seqs, valid_seqs, test_seqs = {}, {}, {}
        for user, seq in self.sequences.items():
            if len(seq) < 3:
                continue
            train_seqs[user] = seq[:-2]
            valid_seqs[user] = seq[:-1]
            test_seqs[user] = seq
        return train_seqs, valid_seqs, test_seqs
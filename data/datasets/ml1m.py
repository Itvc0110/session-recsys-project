from .base_dataset import BaseSequentialDataset

class SequentialDataset(BaseSequentialDataset):
    def __init__(self, data_path, sep='::', max_seq_len=200, min_interactions=5):
        super().__init__(data_path, sep, max_seq_len, min_interactions, is_json=False)
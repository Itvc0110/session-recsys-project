import pytest
import torch
from models.sasrec import SASRec

@pytest.fixture
def model():
    config = {'num_items': 10, 'embedding_size': 4, 'max_seq_len': 5, 'num_layers': 1, 'n_heads': 1, 'inner_size': 4, 'dropout_prob': 0.0, 'device': 'cuda', 'loss_type': 'CE'}
    return SASRec(config)

def test_forward(model):
    item_seq = torch.randint(1, 10, (2, 3))
    item_seq_len = torch.tensor([3, 3])
    output = model.forward(item_seq, item_seq_len)
    assert output.shape == (2, 4)

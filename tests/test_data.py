import pytest
from data.datasets.ml1m import SequentialDataset

@pytest.fixture
def dataset():
    return SequentialDataset('data/raw/ml-1m/ratings.dat', max_seq_len=5)

def test_split(dataset):
    train, valid, test = dataset.split()
    assert len(train) == len(valid) == len(test)
    for u in train:
        assert len(valid[u]) == len(train[u]) + 1
        assert len(test[u]) == len(valid[u]) + 1

import pytest
from src.models.model import GNN

def test_initialization(global_pooling):
    with pytest.raises(ValueError):
        model = GNN(10, 10, 32, embed='global_mean_pol')
import pytest
from src.models.model import GNN

def test_initialization_pooling(global_pooling):
    with pytest.raises(ValueError):
        # Try with invalid pooling
        model = GNN(10, 10, 32, global_pooling='global_mean_pol')
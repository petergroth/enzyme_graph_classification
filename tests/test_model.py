import pytest
from src.models.model import GNN, Classifier

def test_initialization_pooling():
    with pytest.raises(ValueError):
        # Try with invalid pooling
        model = GNN(10, 10, 32, global_pooling='global_mean_pol')

#@pytest.mark.parametrize("n_class_Classifier", [10])
#@pytest.mark.parametrize("n_class_Model", [8])
#def test_initialization_n_classes(n_class_Classifier, n_class_Model):
#    with pytest.raises(ValueError):
#        model = GNN(n_node_features=10, n_classes=n_class_Model)
#        classifier = Classifier(model=model, num_classes=)
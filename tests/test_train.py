import pytorch_lightning as pl
import torch

from src import project_dir
from src.data.enzymes import EnzymesDataModule
from src.models.model import GNN, GraphClassifier
from src.models.train_model import train


def test_train_weights():
    dm = EnzymesDataModule(data_dir=project_dir + "/data/")
    dm.prepare_data()
    model = GNN(n_node_features=dm.num_features, n_classes=dm.num_classes)
    classifier = GraphClassifier(model=model)
    trainer = pl.Trainer(max_epochs=1, checkpoint_callback=False, logger=False)

    init_weights = classifier.model.fc1.weight.detach().clone()
    dm, classifier, trainer = train(dm, classifier, trainer)
    trained_weights = classifier.model.fc1.weight.detach().clone()

    assert not torch.all(torch.eq(init_weights, trained_weights))

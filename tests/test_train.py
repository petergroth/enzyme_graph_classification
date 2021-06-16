import torch
from src.models.train_model import train
from src.models.model import GNN, Classifier
from src.data.enzymes import EnzymesDataModule
import pytorch_lightning as pl
from src import project_dir

class TestTrain:

    def test_train_weights(self):
        dm = EnzymesDataModule(data_dir=project_dir + "/data/")
        dm.prepare_data()
        model = GNN(n_node_features=dm.num_features, n_classes=dm.num_classes)
        classifier = Classifier(model=model)
        trainer = pl.Trainer(max_epochs=1)

        init_weights = classifier.model.fc1.weight.detach().clone()
        dm, classifier, trainer = train(dm, classifier, trainer)
        trained_weights = classifier.model.fc1.weight.detach().clone()

        assert not torch.all(torch.eq(init_weights, trained_weights))

import argparse
from src.models.model import GNN, GraphClassifier
from src.data.enzymes import EnzymesDataModule
from pytorch_lightning.loggers import WandbLogger
from src import project_dir
import pytorch_lightning as pl
import torch

model = GNN(10, 10, 32)
x = torch.rand(100, 10)
edge_index = torch.randint(100, (2, 100))
batch = torch.
model.forward()
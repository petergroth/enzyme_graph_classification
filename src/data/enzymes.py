import numpy as np
import pytorch_lightning as pl
import torch_geometric.transforms as transforms
from torch import manual_seed
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset

from src import project_dir


class EnzymesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir="data/",
        batch_size=64,
        num_workers=0,
        splits=[0.7, 0.15, 0.15],
        seed=42,
    ):
        super(EnzymesDataModule, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.splits = splits
        self.seed = seed
        self.transform = transforms.Compose(
            [
                transforms.NormalizeFeatures(),
            ]
        )

        # Number of graphs, classes and features
        self.num_graphs = 600
        self.num_classes = 6
        self.num_features = 21

    def prepare_data(self):
        # Download data
        TUDataset(
            root=self.data_dir,
            name="ENZYMES",
            use_node_attr=True,
            use_edge_attr=True,
            pre_transform=self.transform,
        )

    def setup(self, stage=None):
        manual_seed(self.seed)
        dataset = TUDataset(
            root=self.data_dir,
            name="ENZYMES",
            use_node_attr=True,
            use_edge_attr=True,
            pre_transform=self.transform,
        ).shuffle()

        split_idx = np.cumsum(
            [int(len(dataset) * prop) for prop in self.splits])
        self.data_train = dataset[: split_idx[0]]
        self.data_val = dataset[split_idx[0]: split_idx[1]]
        self.data_test = dataset[split_idx[1]:]

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("EnzymesDataModule")
        parser.add_argument(
            "--data_dir", default=project_dir + "/data/", type=str)
        parser.add_argument("--batch_size", default=64, type=int)
        parser.add_argument("--num_workers", default=0, type=int)
        parser.add_argument(
            "--splits", default=[0.7, 0.15, 0.15], nargs=3, type=float)
        parser.add_argument("--seed", default=42, type=int)

        return parent_parser

    @staticmethod
    def from_argparse_args(namespace):
        ns_dict = vars(namespace)
        args = {
            "data_dir": ns_dict.get("data_dir", project_dir + "/data/"),
            "batch_size": ns_dict.get("batch_size", 64),
            "num_workers": ns_dict.get("num_workers", 0),
            "splits": ns_dict.get("splits", [0.7, 0.15, 0.15]),
            "seed": ns_dict.get("seed", 42),
        }

        return args


if __name__ == "__main__":
    dm = EnzymesDataModule(data_dir=project_dir + "/data/")
    dm.prepare_data()

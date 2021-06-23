import os

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.loggers import WandbLogger

from src import project_dir
from src.data.enzymes import EnzymesDataModule
from src.models.model import GNN, GraphClassifier


def setup(cfg):
    # Setup seed
    if cfg.train.misc.seed:
        torch.manual_seed(cfg.train.misc.seed)

    # Logger
    wandb_logger = WandbLogger(
        project=cfg.train.misc.wandb_project,
        entity=cfg.train.misc.wandb_entity,
        log_model="all",
        config=cfg,
    )

    # Data
    dm = EnzymesDataModule(**cfg.dataset)
    dm.prepare_data()

    # Model
    model_kwargs = {
        "n_node_features": dm.num_features,
        "n_classes": dm.num_classes,
        **cfg.model,
    }
    model = GNN(**model_kwargs)

    classifier = GraphClassifier(model, lr=cfg.train.misc.lr)
    wandb_logger.watch(classifier)

    # Trainer
    trainer = pl.Trainer(logger=wandb_logger, **cfg.train.trainer)

    return dm, classifier, trainer, model_kwargs


def train(dm, classifier, trainer):
    dm.setup(stage="fit")
    trainer.fit(model=classifier, datamodule=dm)

    return dm, classifier, trainer


def test(dm, classifier, trainer):
    dm.setup(stage="test")
    trainer.test(model=classifier, datamodule=dm)

    return dm, classifier, trainer


@hydra.main(config_path=project_dir + "/conf", config_name="default_config.yaml")
def main(cfg: DictConfig):
    dm, classifier, trainer, model_kwargs = setup(cfg)
    dm, classifier, trainer = train(dm, classifier, trainer)
    torch.save(
        {"model_kwargs": model_kwargs, "state_dict": classifier.model.state_dict()},
        project_dir + cfg.train.misc.model_path,
    )

    dm, classifier, trainer = test(dm, classifier, trainer)

    if cfg.train.misc.azure:
        model_name = os.path.basename(cfg.train.misc.model_path)
        model_file = "outputs/" + model_name
        os.makedirs("outputs", exist_ok=True)

        torch.save(
            {"model_kwargs": model_kwargs, "state_dict": classifier.model.state_dict()},
            model_file,
        )


if __name__ == "__main__":
    main()

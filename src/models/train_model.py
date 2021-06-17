import argparse
from src.models.model import GNN, GraphClassifier
from src.data.enzymes import EnzymesDataModule
from pytorch_lightning.loggers import WandbLogger
from src import project_dir
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


def parser(lightning_class, data_class, model_class):
    """Parses command line."""
    parser = argparse.ArgumentParser()

    # Progam level args
    parser.add_argument('--wandb_project', default='enzymes-test', type=str)
    parser.add_argument(
        '--wandb_entity', default='mlops_enzyme_graph_classification')
    parser.add_argument(
        '--model_dir', default=project_dir + '/models/', type=str)
    parser.add_argument('--azure', action='store_true')
  
    # Training level args
    parser = pl.Trainer.add_argparse_args(parser)

    # Lightning level args
    parser = lightning_class.add_model_specific_args(parser)
    
    # Data level args
    parser = data_class.add_model_specific_args(parser)

    # Model level args
    parser = model_class.add_model_specific_args(parser)

    args = parser.parse_args()

    return args


def setup(args):
    wandb_logger = WandbLogger(
        project=args.wandb_project, entity=args.wandb_entity,
        log_model='all', config=args)

    # Data
    dm = EnzymesDataModule(**EnzymesDataModule.from_argparse_args(args))
    dm.prepare_data()

    # Model
    model = GNN(
        n_node_features=dm.num_features,
        n_classes=dm.num_classes,
        **GNN.from_argparse_args(args))
    classifier = GraphClassifier(
        model, **GraphClassifier.from_argparse_args(args))
    wandb_logger.watch(classifier)

    # Trainer
    checkpoint_callback = ModelCheckpoint(dirpath=args.model_dir)
    trainer = pl.Trainer.from_argparse_args(
        args, logger=wandb_logger, callbacks=[checkpoint_callback])

    return dm, classifier, trainer

def train(dm, classifier, trainer):
    dm.setup(stage='fit')
    trainer.fit(model=classifier, datamodule=dm)

    return dm, classifier, trainer

def test(dm, classifier, trainer):
    dm.setup(stage='test')
    trainer.test(model=classifier, datamodule=dm)

    return dm, classifier, trainer

def main():
    args = parser(GraphClassifier, EnzymesDataModule, GNN)
    dm, classifier, trainer = setup(args)
    dm, classifier, trainer = train(dm, classifier, trainer)
    dm, classifier, trainer = test(dm, classifier, trainer)

if __name__ == '__main__':
    main()

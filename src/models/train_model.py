import argparse
from src.models.model import GNN, GraphClassifier
from src.data.enzymes import EnzymesDataModule
from pytorch_lightning.loggers import WandbLogger
from src import project_dir
import pytorch_lightning as pl

def parser(lightning_class, data_class, model_class):
    """Parses command line."""
    parser = argparse.ArgumentParser()

    # Progam level args
    parser.add_argument('--wandb_project', default='enzymes-test', type=str)
    parser.add_argument(
        '--wandb_entity', default='mlops_enzyme_graph_classification')
    parser.add_argument(
        '--model_path', default=project_dir + '/models/model.pth', type=str)
  
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

def main():
    # Setup
    args = parser(GraphClassifier, EnzymesDataModule, GNN)
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
    trainer = pl.Trainer.from_argparse_args(args, logger=wandb_logger)

    # Train
    dm.setup(stage='fit')
    trainer.fit(model=classifier, datamodule=dm)
    trainer.save_checkpoint(args.model_path)

    # Test
    dm.setup(stage='test')
    trainer.test(datamodule=dm)


if __name__ == '__main__':
    main()

import argparse
from src.models.model import GNN, GraphClassifier
from src.data.enzymes import EnzymesDataModule
from pytorch_lightning.loggers import WandbLogger
from src import project_dir
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import optuna
from optuna.integration import PyTorchLightningPruningCallback

def parser(data_class):
    """Parses command line."""
    parser = argparse.ArgumentParser()

    # Progam level args
    parser.add_argument(
        '--wandb_project', default='enzymes-test-optuna', type=str)
    parser.add_argument(
        '--wandb_entity', default='mlops_enzyme_graph_classification')
    
    # Optimization args
    parser.add_argument('--n_startup_trials', default=5, type=int)
    parser.add_argument('--n_warmup_steps', default=0, type=int)
    parser.add_argument('--n_trials', default=100, type=int)
    parser.add_argument('--timeout', default=None)

    # Training level args
    parser = pl.Trainer.add_argparse_args(parser)
  
    # Data level args
    parser = data_class.add_model_specific_args(parser)

    args = parser.parse_args()

    return args


def suggest_model(trial: optuna.trial.Trial) -> dict:
    hidden_sizes = [trial.suggest_categorical(
        f'hidden_size_{layer}', [32, 64, 128, 256]) for layer in range(2)]

    global_pooling = trial.suggest_categorical(
        'global_pooling',
        ["global_mean_pool", "global_add_pool", "global_max_pool"])

    activation = trial.suggest_categorical(
        'activation',
        ['nn.ReLU', 'nn.LeakyReLU'])

    dropout = trial.suggest_float('dropout', 0, 0.5)

    model_kwargs = {
        'hidden_sizes': hidden_sizes,
        'global_pooling': global_pooling,
        'activation': activation,
        'dropout': dropout,
    }
    
    return model_kwargs


class Objective(object):

    def __init__(self, args):
        self.args = args

    def __call__(self, trial):
        # Hyper parameters
        batch_size = trial.suggest_categorical(
            'batch_size', [8, 16, 32])
        lr = trial.suggest_float('lr', 1e-4, 1e-1)
        model_kwargs = suggest_model(trial)
        
        self.args.batch_size = batch_size
        self.args.lr = lr
        self.args.__dict__.update(model_kwargs)

        # Logger
        wandb_logger = WandbLogger(
            project=self.args.wandb_project, entity=self.args.wandb_entity,
            log_model='all', config=self.args, reinit=True)

        # Data
        dm = EnzymesDataModule(
            **EnzymesDataModule.from_argparse_args(self.args))
        dm.prepare_data()

        # Model 
        model_kwargs['n_node_features'] = dm.num_features
        model_kwargs['n_classes'] = dm.num_classes
        model = GNN(**model_kwargs)
        classifier = GraphClassifier(model, lr=lr)
        wandb_logger.watch(classifier)

        # Pruning 
        pruning_callback = PyTorchLightningPruningCallback(
            trial, monitor="val_Accuracy")

        # Trainer
        trainer = pl.Trainer.from_argparse_args(
            self.args,
            callbacks=[pruning_callback],
            logger=wandb_logger)
        
        dm.setup(stage='fit')
        trainer.fit(model=classifier, datamodule=dm)

        wandb_logger.finalize('0')
        wandb_logger.experiment.finish()

        return trainer.callback_metrics['val_Accuracy'].item()


def main():
    args = parser(EnzymesDataModule)

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=args.n_startup_trials, n_warmup_steps=args.n_warmup_steps)
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(Objective(args), n_trials=args.n_trials, timeout=args.timeout)

    # Print stats
    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == '__main__':
    main()
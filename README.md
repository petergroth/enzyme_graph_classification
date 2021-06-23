enzyme_graph_classification
==============================

Project for the Machine Learning Operations course at DTU (June 2021)

## Train
Use `python3 src/models/train_model.py` to train the `GNN` model if you prefer to specify optional model and training arguments via the command line (uses `argparse` for argument parsing). 

Use `python3 src/models/train_model_hydra.py` to train the `GNN` model if you prefer to specify optional model and training arguments via config files (uses `Hydra` for argument parsing).

 Use `python3 src/models/train_model_optuna.py` to train the `GNN` model and optimize the hyper-parameters of the model (learning rate, batch size, activation function, global pooling method, dropout rate, no. of convolutional channels and size of fully connected layer) with `Optuna`.
 
All training scripts logs training and validation statistics to [Weights & Biases](http://wandb.ai/).  Specify  [Weights & Biases](http://wandb.ai/) project and entity with `--wandb_project` and `--wandb_entity`. Trained models gets saved to `--model_path`.

Optional training arguments are those accepted by [PyTorch Lightning](https://www.pytorchlightning.ai/)'s `Trainer` class ([link](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html)). 

Optional model and data args arguments are:
```
GraphClassifier:
  --lr LR

EnzymesDataModule:
  --data_dir DATA_DIR
  --batch_size BATCH_SIZE
  --num_workers NUM_WORKERS
  --splits SPLITS SPLITS SPLITS
  --seed SEED

GNN:
  --conv_channels CONV_CHANNELS
  --fc_size FC_SIZE
  --global_pooling {global_mean_pool,global_add_pool,global_max_pool}
  --activation {nn.ReLU,nn.Tanh,nn.RReLU,nn.LeakyReLU,nn.ELU}
  --dropout DROPOUT```
 
## Predict

## Microsoft Azure Machine Learning

The project has succesfully been implemented via Microsoft Azure Machine Learning (Azure ML). This allows for both training and deployment.

### Usage:
The repository is cloned into an Azure workspace via `git clone https://github.com/petergroth/enzyme_graph_classification.git`. 

#### Environment creation
To successfully train and deploy a model, an environment has to be created. This is done via `azure_deployment/create_env.py` which takes the argument `--train` or `--deploy` to create environments for training and deployment, respectively. (These cannot be identical due to a dependency issue, where the deployment requires the `azureml-defaults` package to be installed. This requires a specific version of  the `configparser` module. The chosen logging framework, `wandb`, requires a conflicting version of `configparser` which is newer than the required version for `azureml-defaults`.)  
The created environments will be named `EGC_train` and `EGC_deploy`.

Example usage (from Azure):

`python azure_deployment/create_env.py --train`

#### Training
A model is trained via `azure_deployment/train_and_register.py`, where the model name and Azure ML compute targets have to be defined in the script as well as the hyperparemeters for the model. After training, the model is registered to the Azure ML workspace. 
Note: in order to successfully log the training run via `wandb`, the API-key has to be defined. To do this, create a script named `azure_deployment/wandb_api_key.py`, in which the Python variable `WANDB_API_KEY` is defined as a string equal to the users API-key.

Example usage (from Azure):

`python azure_deployment/train_and_register.py`

#### Deployment
The trained and registered model is deployed via `azure_deployment/deploy_model.py`, where the model and service names have to be specified manually. The deployment uses the scoring script `azure_deployment/azure_scoring_script.py`.

Example usage (from Azure):

`python azure_deployment/deploy_model.py`

#### Inference 
The deployed model can be used for prediction via `src/models/predict_with_azure_service.py`. This requires 3 arguments. The two first specifies the `edge_table_file` and the `node_attributes_file`, while the third argument is the provided `scoring_uri`, which is printed after successful deployment with Azure. Examples of the two data files can be found in `data/processed/single_graphs`.

Example usage (locally): 

`python src/model/predict_with_azure_service.py data/processed/single_graphs/graph_100_edges.txt data/processed/single_graphs/graph_100_node_attributes.txt scoring_uri` 

which prints the logits, probabilites, and label of the prediction.


#### Hyperparameter optimization
Optuna can be used to optimize the hyperparameter selection. To do this, the `azure_deployment/optimize_hparams.py` script can be used. The Optuna details are specified in the script, which submits the `src/models/train_model_optuna.py` script to the specified Azure ML compute target. The optimization can be tracked directly via `wandb`.

Example usage (from Azure):

`python azure_deployment/optimize_hparams.py`

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

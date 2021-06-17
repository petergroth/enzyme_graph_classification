import json
import joblib
import numpy as np
from azureml.core.model import Model
import torch

class Batch:
    def __init__(self, x, edge_index, batch):
        self.x = x
        self.edge_index = edge_index
        self.batch = batch

# Called when the service is loaded
def init():
    global model
    # Get the path to the deployed model file and load it
    model_path = Model.get_model_path('SOME_MODEL_NAME')
    model = joblib.load(model_path)

# Called when a request is received
def run(raw_data):
    # Get the input data as tensors
    input_json = json.loads(raw_data)
    x = torch.Tensor(input_json['x'])
    edge_index = torch.Tensor(input_json['edge_index'])
    batch = torch.Tensor(input_json['batch'])

    # Prepare data to model
    batch = Batch(x=x, edge_index=edge_index, batch=batch)

    # Get a prediction from the model
    predictions = model.forward(batch)

    predictions = predictions.numpy().tolist()

    return json.dumps(predictions)
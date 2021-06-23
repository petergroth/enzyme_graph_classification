import json

import torch

from azureml.core.model import Model
from src.models.model import GNN


# Called when the service is loaded
def init():
    global model
    # Get the path to the deployed model file and load it
    model_path = Model.get_model_path("debug_model.ckpt")
    model_dict = torch.load(model_path)
    model = GNN(**model_dict["model_kwargs"])
    model.load_state_dict(model_dict["state_dict"])
    model.eval()


# Called when a request is received
def run(raw_data):
    # Get the input data as tensors
    input_json = json.loads(raw_data)
    x = torch.Tensor(input_json["x"]).to(torch.float)
    edge_index = torch.Tensor(input_json["edge_index"]).to(torch.int64)
    batch = torch.Tensor(input_json["batch"]).to(torch.int64)

    # Get a prediction from the model
    predictions = model.forward(x, edge_index, batch).detach().squeeze()
    predictions = predictions.numpy().tolist()

    return json.dumps(predictions)

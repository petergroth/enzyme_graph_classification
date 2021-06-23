# run-azure.py
import os

from azureml.core import Environment, Experiment, ScriptRunConfig, Workspace
from wandb_api_key import WANDB_API_KEY

if __name__ == "__main__":
    # Specify model name. Must agree with name in train_model.py
    model_name = "debug_model.ckpt"
    compute_target = "compute-v1"

    # Setup experiment
    ws = Workspace.from_config()
    experiment = Experiment(workspace=ws, name="EGC_training_run")

    # Setup environment
    env = Environment.get(workspace=ws, name="EGC_train")
    env.environment_variables = {"WANDB_API_KEY": WANDB_API_KEY}

    # Define configuration file
    config = ScriptRunConfig(
        source_directory="./src/models/",
        script="train_model.py",
        arguments=[
            "--max_steps",
            1,
            "--conv_channels",
            64,
            "--fc_size",
            64,
            "--batch_size",
            16,
            "--global_pooling",
            "global_max_pool",
            "--activation",
            "nn.ReLU" "-azure",
        ],
        environment=env,
        compute_target=compute_target,
    )

    # Submit experiment and wait for completion
    run = experiment.submit(config)
    run.wait_for_completion()

    # Save resulting model and register
    os.makedirs("./outputs", exist_ok=True)
    run.download_file(
        name="outputs/" + model_name, output_file_path="./outputs/" + model_name
    )

    run.register_model(model_path="outputs/" + model_name, model_name=model_name)
